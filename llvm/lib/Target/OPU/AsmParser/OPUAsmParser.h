//===-- OPUAsmParser.cpp - Parse OPU assembly to MCInst instructions --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "OPU.h"
#include "OPUKernelCodeT.h"
#include "OPUDefines.h"
#include "OPUInstrInfo.h"
#include "Utils/OPUAsmUtils.h"
#include "Utils/OPUBaseInfo.h"
#include "Utils/OPUKernelCodeTUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/OPUMetadata.h"
#include "llvm/Support/OPUKernelDescriptor.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MachineValueType.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"


#include "MCTargetDesc/OPUAsmBackend.h"
#include "MCTargetDesc/OPUMCExpr.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "MCTargetDesc/OPUTargetStreamer.h"
#include "TargetInfo/OPUTargetInfo.h"
#include "Utils/OPUBaseInfo.h"
#include "Utils/OPUMatInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"

#include <limits>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <string>


using namespace llvm;
using namespace llvm::OPU;
using namespace llvm::pps;

// Include the auto-generated portion of the compress emitter.
// #define GEN_COMPRESS_INSTR
// #include "OPUGenCompressInstEmitter.inc"

namespace {

enum RegisterKind { IS_UNKNOWN, IS_VGPR, IS_SGPR, IS_AGPR, IS_TTMP, IS_SPECIAL };

// Holds info related to the current kernel, e.g. count of SGPRs used.
// Kernel scope begins at .amdgpu_hsa_kernel directive, ends at next
// .amdgpu_hsa_kernel or at EOF.
class KernelScopeInfo {
  int SgprIndexUnusedMin = -1;
  int VgprIndexUnusedMin = -1;
  MCContext *Ctx = nullptr;

  void usesSgprAt(int i) {
    if (i >= SgprIndexUnusedMin) {
      SgprIndexUnusedMin = ++i;
      if (Ctx) {
        MCSymbol * const Sym = Ctx->getOrCreateSymbol(Twine(".kernel.sgpr_count"));
        Sym->setVariableValue(MCConstantExpr::create(SgprIndexUnusedMin, *Ctx));
      }
    }
  }

  void usesVgprAt(int i) {
    if (i >= VgprIndexUnusedMin) {
      VgprIndexUnusedMin = ++i;
      if (Ctx) {
        MCSymbol * const Sym = Ctx->getOrCreateSymbol(Twine(".kernel.vgpr_count"));
        Sym->setVariableValue(MCConstantExpr::create(VgprIndexUnusedMin, *Ctx));
      }
    }
  }

public:
  KernelScopeInfo() = default;

  void initialize(MCContext &Context) {
    Ctx = &Context;
    usesSgprAt(SgprIndexUnusedMin = -1);
    usesVgprAt(VgprIndexUnusedMin = -1);
  }

  void usesRegister(RegisterKind RegKind, unsigned DwordRegIndex, unsigned RegWidth) {
    switch (RegKind) {
      case IS_SGPR: usesSgprAt(DwordRegIndex + RegWidth - 1); break;
      case IS_AGPR: // fall through
      case IS_VGPR: usesVgprAt(DwordRegIndex + RegWidth - 1); break;
      default: break;
    }
  }
};

class OPUAsmParser;

// static bool OPUAsmParser::classifySymbolRef(const MCExpr *Expr,
//                                 OPUMCExpr::VariantKind &Kind,
//                                 int64_t &Addend);


class OPUOperand : public MCParsedAsmOperand {
  enum class KindTy {
    Token,
    Immediate,
    Register,
    Expression,
    SystemRegister
  } Kind;

  bool IsRV64;

  SMLoc StartLoc, EndLoc;

  const OPUAsmParser *AsmParser;

public:
  OPUOperand(KindTy Kind_, const OPUAsmParser *AsmParser_ = nullptr)
    : MCParsedAsmOperand(), Kind(Kind_), AsmParser(AsmParser_) {}

  OPUOperand(const OPUOperand &o) : MCParsedAsmOperand() {
    Kind = o.Kind;
    IsRV64 = o.IsRV64;
    StartLoc = o.StartLoc;
    EndLoc = o.EndLoc;
    switch (Kind) {
    case KindTy::Register:
      Reg = o.Reg;
      break;
    case KindTy::Immediate:
      Imm = o.Imm;
      break;
    case KindTy::Token:
      Tok = o.Tok;
      break;
    case KindTy::SystemRegister:
      SysReg = o.SysReg;
      break;
    case KindTy::Expression:
      llvm_unreachable("OPUOperand ctr wip");
      break;
    }
  }

  using Ptr = std::unique_ptr<OPUOperand>;

  struct Modifiers {
    bool Abs = false;
    bool Neg = false;
    bool Sext = false;

    bool hasFPModifiers() const { return Abs || Neg; }
    bool hasIntModifiers() const { return Sext; }
    bool hasModifiers() const { return hasFPModifiers() || hasIntModifiers(); }

    int64_t getFPModifiersOperand() const {
      int64_t Operand = 0;
      Operand |= Abs ? OPUSrcMods::ABS : 0u;
      Operand |= Neg ? OPUSrcMods::NEG : 0u;
      return Operand;
    }

    int64_t getIntModifiersOperand() const {
      int64_t Operand = 0;
      Operand |= Sext ? OPUSrcMods::SEXT : 0u;
      return Operand;
    }

    int64_t getModifiersOperand() const {
      assert(!(hasFPModifiers() && hasIntModifiers())
           && "fp and int modifiers should not be used simultaneously");
      if (hasFPModifiers()) {
        return getFPModifiersOperand();
      } else if (hasIntModifiers()) {
        return getIntModifiersOperand();
      } else {
        return 0;
      }
    }

    friend raw_ostream &operator <<(raw_ostream &OS, OPUOperand::Modifiers Mods);
  };

  enum ImmTy {
    ImmTyNone,
    ImmTyGDS,
    ImmTyLDS,
    ImmTyOffen,
    ImmTyIdxen,
    ImmTyAddr64,
    ImmTyOffset,
    ImmTyInstOffset,
    ImmTyOffset0,
    ImmTyOffset1,
    ImmTyDLC,
    ImmTyGLC,
    ImmTySLC,
    ImmTyTFE,
    ImmTyD16,
    ImmTyClampSI,
    ImmTyOModSI,
    ImmTyDPP8,
    ImmTyDppCtrl,
    ImmTyDppRowMask,
    ImmTyDppBankMask,
    ImmTyDppBoundCtrl,
    ImmTyDppFi,
    ImmTySdwaDstSel,
    ImmTySdwaSrc0Sel,
    ImmTySdwaSrc1Sel,
    ImmTySdwaDstUnused,
    ImmTyDMask,
    ImmTyDim,
    ImmTyUNorm,
    ImmTyDA,
    ImmTyR128A16,
    ImmTyLWE,
    ImmTyExpTgt,
    ImmTyExpCompr,
    ImmTyExpVM,
    ImmTyFORMAT,
    ImmTyHwreg,
    ImmTyOff,
    ImmTySendMsg,
    ImmTyInterpSlot,
    ImmTyInterpAttr,
    ImmTyAttrChan,
    ImmTyOpSel,
    ImmTyOpSelHi,
    ImmTyNegLo,
    ImmTyNegHi,
    ImmTySwizzle,
    ImmTyGprIdxMode,
    ImmTyHigh,
    ImmTyBLGP,
    ImmTyCBSZ,
    ImmTyABID,
    ImmTyEndpgm,
  };

// private:

  struct TokOp {
    const char *Data;
    unsigned Length;
  };

  struct ImmOp {
    union {
    int64_t Int;
    const llvm::MCExpr *MCExpr;
    } Val;
    ImmTy Type;
    bool IsFPImm;
    Modifiers Mods;
  };
  /*
  struct ImmOp {
    const MCExpr *Val;
  };
  */
  struct RegOp {
    Register RegNum;
  };

  struct RegOp_ppt {
    unsigned RegNo;
    Modifiers Mods;
  };

  struct SysRegOp {
    const char *Data;
    unsigned Length;
    unsigned Encoding;
    // FIXME: Add the Encoding parsed fields as needed for checks,
    // e.g.: read/write or user/supervisor/machine privileges.
  };


  union {
    TokOp Tok;
    StringRef Tok_StringRef;
    ImmOp Imm;
    RegOp Reg;
    RegOp_ppt Reg_ppt;
    const MCExpr *Expr;
    struct SysRegOp SysReg;
  };

public:
  bool isToken() const override {
    if (Kind == KindTy::Token)
      return true;

    // When parsing operands, we can't always tell if something was meant to be
    // a token, like 'gds', or an expression that references a global variable.
    // In this case, we assume the string is an expression, and if we need to
    // interpret is a token, then we treat the symbol name as the token.
    return isSymbolRefExpr();
  }

  bool isSystemRegister() const { return Kind == KindTy::SystemRegister; }

  bool isSymbolRefExpr() const {
    return isExpr() && Expr && isa<MCSymbolRefExpr>(Expr);
  }

  bool isImm() const override {
    return Kind == KindTy::Immediate;
  }

  bool isInlinableImm(MVT type) const;
  bool isLiteralImm(MVT type) const;

  bool isRegKind() const {
    return Kind == KindTy::Register;
  }

  bool isReg() const override {
    return isRegKind() && !hasModifiers();
  }

  bool isRegOrImmWithInputMods(unsigned RCID, MVT type) const {
    return isRegClass(RCID) || isInlinableImm(type) || isLiteralImm(type);
  }

  bool isRegOrImmWithInt16InputMods() const {
    return isRegOrImmWithInputMods(OPU::VS_32RegClassID, MVT::i16);
  }

  bool isRegOrImmWithInt32InputMods() const {
    return isRegOrImmWithInputMods(OPU::VS_32RegClassID, MVT::i32);
  }

  bool isRegOrImmWithInt64InputMods() const {
    return isRegOrImmWithInputMods(OPU::VS_64RegClassID, MVT::i64);
  }

  bool isRegOrImmWithFP16InputMods() const {
    return isRegOrImmWithInputMods(OPU::VS_32RegClassID, MVT::f16);
  }

  bool isRegOrImmWithFP32InputMods() const {
    return isRegOrImmWithInputMods(OPU::VS_32RegClassID, MVT::f32);
  }

  bool isRegOrImmWithFP64InputMods() const {
    return isRegOrImmWithInputMods(OPU::VS_64RegClassID, MVT::f64);
  }

  bool isVReg() const {
    return isRegClass(OPU::VPR_32RegClassID) ||
           isRegClass(OPU::VReg_64RegClassID) ||
           isRegClass(OPU::VReg_96RegClassID) ||
           isRegClass(OPU::VReg_128RegClassID); /*||
           isRegClass(OPU::VReg_160RegClassID) ||
           isRegClass(OPU::VReg_256RegClassID) ||
           isRegClass(OPU::VReg_512RegClassID) ||
           isRegClass(OPU::VReg_1024RegClassID);*/
  }

  bool isVReg32() const {
    return isRegClass(OPU::VPR_32RegClassID);
  }

  bool isVReg32OrOff() const {
    return isOff() || isVReg32();
  }

  bool isSDWAOperand(MVT type) const;
  bool isSDWAFP16Operand() const;
  bool isSDWAFP32Operand() const;
  bool isSDWAInt16Operand() const;
  bool isSDWAInt32Operand() const;

  bool isImmTy(ImmTy ImmT) const {
    return isImm() && Imm.Type == ImmT;
  }

  bool isImmModifier() const {
    return isImm() && Imm.Type != ImmTyNone;
  }

  bool isClampSI() const { return isImmTy(ImmTyClampSI); }
  bool isOModSI() const { return isImmTy(ImmTyOModSI); }
  bool isDMask() const { return isImmTy(ImmTyDMask); }
  bool isDim() const { return isImmTy(ImmTyDim); }
  bool isUNorm() const { return isImmTy(ImmTyUNorm); }
  bool isDA() const { return isImmTy(ImmTyDA); }
  bool isR128A16() const { return isImmTy(ImmTyR128A16); }
  bool isLWE() const { return isImmTy(ImmTyLWE); }
  bool isOff() const { return isImmTy(ImmTyOff); }
  bool isExpTgt() const { return isImmTy(ImmTyExpTgt); }
  bool isExpVM() const { return isImmTy(ImmTyExpVM); }
  bool isExpCompr() const { return isImmTy(ImmTyExpCompr); }
  bool isOffen() const { return isImmTy(ImmTyOffen); }
  bool isIdxen() const { return isImmTy(ImmTyIdxen); }
  bool isAddr64() const { return isImmTy(ImmTyAddr64); }
  bool isOffset() const { return isImmTy(ImmTyOffset) && isUInt<16>(getImmVal()); }
  bool isOffset0() const { return isImmTy(ImmTyOffset0) && isUInt<8>(getImmVal()); }
  bool isOffset1() const { return isImmTy(ImmTyOffset1) && isUInt<8>(getImmVal()); }

  bool isFlatOffset() const { return isImmTy(ImmTyOffset) || isImmTy(ImmTyInstOffset); }
  bool isGDS() const { return isImmTy(ImmTyGDS); }
  bool isLDS() const { return isImmTy(ImmTyLDS); }
  bool isDLC() const { return isImmTy(ImmTyDLC); }
  bool isGLC() const { return isImmTy(ImmTyGLC); }
  bool isSLC() const { return isImmTy(ImmTySLC); }
  bool isTFE() const { return isImmTy(ImmTyTFE); }
  bool isD16() const { return isImmTy(ImmTyD16); }
  bool isFORMAT() const { return isImmTy(ImmTyFORMAT) && isUInt<8>(getImmVal()); }
  bool isBankMask() const { return isImmTy(ImmTyDppBankMask); }
  bool isRowMask() const { return isImmTy(ImmTyDppRowMask); }
  bool isBoundCtrl() const { return isImmTy(ImmTyDppBoundCtrl); }
  bool isFI() const { return isImmTy(ImmTyDppFi); }
  bool isSDWADstSel() const { return isImmTy(ImmTySdwaDstSel); }
  bool isSDWASrc0Sel() const { return isImmTy(ImmTySdwaSrc0Sel); }
  bool isSDWASrc1Sel() const { return isImmTy(ImmTySdwaSrc1Sel); }
  bool isSDWADstUnused() const { return isImmTy(ImmTySdwaDstUnused); }
  bool isInterpSlot() const { return isImmTy(ImmTyInterpSlot); }
  bool isInterpAttr() const { return isImmTy(ImmTyInterpAttr); }
  bool isAttrChan() const { return isImmTy(ImmTyAttrChan); }
  bool isOpSel() const { return isImmTy(ImmTyOpSel); }
  bool isOpSelHi() const { return isImmTy(ImmTyOpSelHi); }
  bool isNegLo() const { return isImmTy(ImmTyNegLo); }
  bool isNegHi() const { return isImmTy(ImmTyNegHi); }
  bool isHigh() const { return isImmTy(ImmTyHigh); }

  bool isMod() const {
    return isClampSI() || isOModSI();
  }

  bool isRegOrImm() const {
    return isReg() || isImm();
  }

  bool isRegClass(unsigned RCID) const;

  bool isInlineValue() const;

  bool isRegOrInlineNoMods(unsigned RCID, MVT type) const {
    return (isRegClass(RCID) || isInlinableImm(type)) && !hasModifiers();
  }

  bool isSCSrcB16() const {
    return isRegOrInlineNoMods(OPU::SReg_32RegClassID, MVT::i16);
  }

  // TODO i add it for missing isSSrcB8
  bool isSCSrcB8() const {
    return isRegOrInlineNoMods(OPU::SReg_32RegClassID, MVT::i8);
  }

  bool isSCSrcV2B8() const {
    return isSCSrcB8();
  }

  bool isSCSrcV4B8() const {
    return isSCSrcB8();
  }

  bool isSCSrcV2B16() const {
    return isSCSrcB16();
  }

  bool isSCSrcB32() const {
    return isRegOrInlineNoMods(OPU::SReg_32RegClassID, MVT::i32);
  }

  bool isSCSrcB64() const {
    return isRegOrInlineNoMods(OPU::SReg_64RegClassID, MVT::i64);
  }

  bool isBoolReg() const;

  bool isSCSrcF16() const {
    return isRegOrInlineNoMods(OPU::SReg_32RegClassID, MVT::f16);
  }

  bool isSCSrcV2F16() const {
    return isSCSrcF16();
  }

  bool isSCSrcV4F16() const {
    return isSCSrcF16();
  }

  bool isSCSrcF32() const {
    return isRegOrInlineNoMods(OPU::SReg_32RegClassID, MVT::f32);
  }

  bool isSCSrcF64() const {
    return isRegOrInlineNoMods(OPU::SReg_64RegClassID, MVT::f64);
  }

  bool isSSrcB32() const {
    return isSCSrcB32() || isLiteralImm(MVT::i32) || isExpr();
  }

  bool isSSrcB16() const {
    return isSCSrcB16() || isLiteralImm(MVT::i16);
  }

  bool isSSrcB8() const {
    return isSCSrcB8() || isLiteralImm(MVT::i8);
  }

  bool isSSrcV2B16() const {
    llvm_unreachable("cannot happen");
    return isSSrcB16();
  }

  bool isSSrcV2B8() const {
    llvm_unreachable("cannot happen");
    return isSSrcB8();
  }

  bool isSSrcV4B8() const {
    llvm_unreachable("cannot happen");
    return isSSrcB8();
  }

  bool isSSrcB64() const {
    // TODO: Find out how SALU supports extension of 32-bit literals to 64 bits.
    // See isVSrc64().
    return isSCSrcB64() || isLiteralImm(MVT::i64);
  }

  bool isSSrcF32() const {
    return isSCSrcB32() || isLiteralImm(MVT::f32) || isExpr();
  }

  bool isSSrcF64() const {
    return isSCSrcB64() || isLiteralImm(MVT::f64);
  }

  bool isSSrcF16() const {
    return isSCSrcB16() || isLiteralImm(MVT::f16);
  }

  bool isSSrcV2F16() const {
    llvm_unreachable("cannot happen");
    return isSSrcF16();
  }

  bool isSSrcV4F16() const {
    llvm_unreachable("cannot happen");
    return isSSrcF16();
  }

  bool isSSrcOrLdsB32() const {
    return isRegOrInlineNoMods(OPU::SRegOrLds_32RegClassID, MVT::i32) ||
           isLiteralImm(MVT::i32) || isExpr();
  }

  bool isVCSrcB32() const {
    return isRegOrInlineNoMods(OPU::VS_32RegClassID, MVT::i32);
  }

  bool isVCSrcB64() const {
    return isRegOrInlineNoMods(OPU::VS_64RegClassID, MVT::i64);
  }

  bool isVCSrcB16() const {
    return isRegOrInlineNoMods(OPU::VS_32RegClassID, MVT::i16);
  }

  bool isVCSrcB8() const {
    return isRegOrInlineNoMods(OPU::VS_32RegClassID, MVT::i8);
  }

  bool isVCSrcV2B16() const {
    return isVCSrcB16();
  }

  bool isVCSrcV4B8() const {
    return isVCSrcB8();
  }

  bool isVCSrcF32() const {
    return isRegOrInlineNoMods(OPU::VS_32RegClassID, MVT::f32);
  }

  bool isVCSrcF64() const {
    return isRegOrInlineNoMods(OPU::VS_64RegClassID, MVT::f64);
  }

  bool isVCSrcF16() const {
    return isRegOrInlineNoMods(OPU::VS_32RegClassID, MVT::f16);
  }

  bool isVCSrcV2F16() const {
    return isVCSrcF16();
  }

  bool isVCSrcV4F16() const {
    return isVCSrcF16();
  }

  bool isVSrcB32() const {
    return isVCSrcF32() || isLiteralImm(MVT::i32) || isExpr();
  }

  bool isVSrcB64() const {
    return isVCSrcF64() || isLiteralImm(MVT::i64);
  }

  bool isVSrcB16() const {
    return isVCSrcF16() || isLiteralImm(MVT::i16);
  }

  bool isVSrcB8() const {
    return isVCSrcB8() || isLiteralImm(MVT::i8);
  }

  bool isVSrcV2B16() const {
    return isVSrcB16() || isLiteralImm(MVT::v2i16);
  }

  bool isVSrcV4B8() const {
    return isVSrcB8() || isLiteralImm(MVT::v4i8);
  }

  bool isVSrcF32() const {
    return isVCSrcF32() || isLiteralImm(MVT::f32) || isExpr();
  }

  bool isVSrcF64() const {
    return isVCSrcF64() || isLiteralImm(MVT::f64);
  }

  bool isVSrcF16() const {
    return isVCSrcF16() || isLiteralImm(MVT::f16);
  }

  bool isVSrcV2F16() const {
    return isVSrcF16() || isLiteralImm(MVT::v2f16);
  }

  bool isVSrcV4F16() const {
    return isVSrcF16() || isLiteralImm(MVT::v4f16);
  }

  bool isVISrcB32() const {
    return isRegOrInlineNoMods(OPU::VPR_32RegClassID, MVT::i32);
  }

  bool isVISrcB16() const {
    return isRegOrInlineNoMods(OPU::VPR_32RegClassID, MVT::i16);
  }

  bool isVISrcV2B16() const {
    return isVISrcB16();
  }

  bool isVISrcF32() const {
    return isRegOrInlineNoMods(OPU::VPR_32RegClassID, MVT::f32);
  }

  bool isVISrcF16() const {
    return isRegOrInlineNoMods(OPU::VPR_32RegClassID, MVT::f16);
  }

  bool isVISrcV2F16() const {
    return isVISrcF16() || isVISrcB32();
  }
/*
  bool isAISrcB32() const {
    return isRegOrInlineNoMods(OPU::AGPR_32RegClassID, MVT::i32);
  }

  bool isAISrcB16() const {
    return isRegOrInlineNoMods(OPU::AGPR_32RegClassID, MVT::i16);
  }

  bool isAISrcV2B16() const {
    return isAISrcB16();
  }

  bool isAISrcF32() const {
    return isRegOrInlineNoMods(OPU::AGPR_32RegClassID, MVT::f32);
  }

  bool isAISrcF16() const {
    return isRegOrInlineNoMods(OPU::AGPR_32RegClassID, MVT::f16);
  }

  bool isAISrcV2F16() const {
    return isAISrcF16() || isAISrcB32();
  }

  bool isAISrc_128B32() const {
    return isRegOrInlineNoMods(OPU::AReg_128RegClassID, MVT::i32);
  }

  bool isAISrc_128B16() const {
    return isRegOrInlineNoMods(OPU::AReg_128RegClassID, MVT::i16);
  }

  bool isAISrc_128V2B16() const {
    return isAISrc_128B16();
  }

  bool isAISrc_128F32() const {
    return isRegOrInlineNoMods(OPU::AReg_128RegClassID, MVT::f32);
  }

  bool isAISrc_128F16() const {
    return isRegOrInlineNoMods(OPU::AReg_128RegClassID, MVT::f16);
  }

  bool isAISrc_128V2F16() const {
    return isAISrc_128F16() || isAISrc_128B32();
  }

  bool isAISrc_512B32() const {
    return isRegOrInlineNoMods(OPU::AReg_512RegClassID, MVT::i32);
  }

  bool isAISrc_512B16() const {
    return isRegOrInlineNoMods(OPU::AReg_512RegClassID, MVT::i16);
  }

  bool isAISrc_512V2B16() const {
    return isAISrc_512B16();
  }

  bool isAISrc_512F32() const {
    return isRegOrInlineNoMods(OPU::AReg_512RegClassID, MVT::f32);
  }

  bool isAISrc_512F16() const {
    return isRegOrInlineNoMods(OPU::AReg_512RegClassID, MVT::f16);
  }

  bool isAISrc_512V2F16() const {
    return isAISrc_512F16() || isAISrc_512B32();
  }

  bool isAISrc_1024B32() const {
    return isRegOrInlineNoMods(OPU::AReg_1024RegClassID, MVT::i32);
  }

  bool isAISrc_1024B16() const {
    return isRegOrInlineNoMods(OPU::AReg_1024RegClassID, MVT::i16);
  }

  bool isAISrc_1024V2B16() const {
    return isAISrc_1024B16();
  }

  bool isAISrc_1024F32() const {
    return isRegOrInlineNoMods(OPU::AReg_1024RegClassID, MVT::f32);
  }

  bool isAISrc_1024F16() const {
    return isRegOrInlineNoMods(OPU::AReg_1024RegClassID, MVT::f16);
  }

  bool isAISrc_1024V2F16() const {
    return isAISrc_1024F16() || isAISrc_1024B32();
  }
*/
  bool isKImmFP32() const {
    return isLiteralImm(MVT::f32);
  }

  bool isKImmFP16() const {
    return isLiteralImm(MVT::f16);
  }

  bool isMem() const override {
    return false;
  }

  bool isExpr() const {
    return Kind == KindTy::Expression;
  }

  bool isSoppBrTarget() const {
    return isExpr() || isImm();
  }

  bool isSWaitCnt() const;
  bool isHwreg() const;
  bool isSendMsg() const;
  bool isSwizzle() const;
  bool isSMRDOffset8() const;
  bool isSMRDOffset20() const;
  bool isSMRDLiteralOffset() const;
  bool isDPP8() const;
  bool isDPPCtrl() const;
  bool isBLGP() const;
  bool isCBSZ() const;
  bool isABID() const;
  bool isVPRIdxMode() const;
  bool isS16Imm() const;
  bool isU16Imm() const;
  bool isEndpgm() const;

  StringRef getExpressionAsToken() const {
    assert(isExpr());
    const MCSymbolRefExpr *S = cast<MCSymbolRefExpr>(Expr);
    return S->getSymbol().getName();
  }

  StringRef getToken() const {
    assert(isToken());

    if (Kind == KindTy::Expression)
      return getExpressionAsToken();

    return StringRef(Tok.Data, Tok.Length);
  }

  int64_t getImmVal() const {
    assert(isImm());
    return Imm.Val.Int;
  }

  ImmTy getImmTy() const {
    assert(isImm());
    return Imm.Type;
  }

  // FIXME should we need to return Reg_ppt.RegNo
  unsigned getReg() const override {
    assert(isRegKind());
    // if (isPPT()) {
    //     return Reg_ppt.RegNo;
    // } else {
        return Reg.RegNum.id();
    // }
  }
/*
  unsigned getReg() const override {
    assert(Kind == KindTy::Register && "Invalid type access!");
    return Reg.RegNum.id();
  }
  */

  SMRange getLocRange() const {
    return SMRange(StartLoc, EndLoc);
  }

  Modifiers getModifiers() const {
    assert(isRegKind() || isImmTy(ImmTyNone));
    return isRegKind() ? Reg_ppt.Mods : Imm.Mods;
  }

  void setModifiers(Modifiers Mods) {
    assert(isRegKind() || isImmTy(ImmTyNone));
    if (isRegKind())
      Reg_ppt.Mods = Mods;
    else
      Imm.Mods = Mods;
  }

  bool hasModifiers() const {
    return getModifiers().hasModifiers();
  }

  bool hasFPModifiers() const {
    return getModifiers().hasFPModifiers();
  }

  bool hasIntModifiers() const {
    return getModifiers().hasIntModifiers();
  }

  uint64_t applyInputFPModifiers(uint64_t Val, unsigned Size) const;

  void addImmOperands(MCInst &Inst, unsigned N, bool ApplyModifiers = true) const;

  void addLiteralImmOperand(MCInst &Inst, int64_t Val, bool ApplyModifiers) const;

  template <unsigned Bitwidth>
  void addKImmFPOperands(MCInst &Inst, unsigned N) const;

  void addKImmFP16Operands(MCInst &Inst, unsigned N) const {
    addKImmFPOperands<16>(Inst, N);
  }

  void addKImmFP32Operands(MCInst &Inst, unsigned N) const {
    addKImmFPOperands<32>(Inst, N);
  }

  void addRegOperands(MCInst &Inst, unsigned N) const;

  void addBoolRegOperands(MCInst &Inst, unsigned N) const {
    addRegOperands(Inst, N);
  }

  void addRegOrImmOperands(MCInst &Inst, unsigned N) const {
    if (isRegKind())
      addRegOperands(Inst, N);
    else if (isExpr())
      Inst.addOperand(MCOperand::createExpr(Expr));
    else
      addImmOperands(Inst, N);
  }

  void addRegOrImmWithInputModsOperands(MCInst &Inst, unsigned N) const {
    Modifiers Mods = getModifiers();
    Inst.addOperand(MCOperand::createImm(Mods.getModifiersOperand()));
    if (isRegKind()) {
      addRegOperands(Inst, N);
    } else {
      addImmOperands(Inst, N, false);
    }
  }

  void addRegOrImmWithFPInputModsOperands(MCInst &Inst, unsigned N) const {
    assert(!hasIntModifiers());
    addRegOrImmWithInputModsOperands(Inst, N);
  }

  void addRegOrImmWithIntInputModsOperands(MCInst &Inst, unsigned N) const {
    assert(!hasFPModifiers());
    addRegOrImmWithInputModsOperands(Inst, N);
  }

  void addRegWithInputModsOperands(MCInst &Inst, unsigned N) const {
    Modifiers Mods = getModifiers();
    Inst.addOperand(MCOperand::createImm(Mods.getModifiersOperand()));
    assert(isRegKind());
    addRegOperands(Inst, N);
  }

  void addRegWithFPInputModsOperands(MCInst &Inst, unsigned N) const {
    assert(!hasIntModifiers());
    addRegWithInputModsOperands(Inst, N);
  }

  void addRegWithIntInputModsOperands(MCInst &Inst, unsigned N) const {
    assert(!hasFPModifiers());
    addRegWithInputModsOperands(Inst, N);
  }

  void addSoppBrTargetOperands(MCInst &Inst, unsigned N) const {
    if (isImm())
      addImmOperands(Inst, N);
    else {
      assert(isExpr());
      Inst.addOperand(MCOperand::createExpr(Expr));
    }
  }

  static void printImmTy(raw_ostream& OS, ImmTy Type) {
    switch (Type) {
    case ImmTyNone: OS << "None"; break;
    case ImmTyGDS: OS << "GDS"; break;
    case ImmTyLDS: OS << "LDS"; break;
    case ImmTyOffen: OS << "Offen"; break;
    case ImmTyIdxen: OS << "Idxen"; break;
    case ImmTyAddr64: OS << "Addr64"; break;
    case ImmTyOffset: OS << "Offset"; break;
    case ImmTyInstOffset: OS << "InstOffset"; break;
    case ImmTyOffset0: OS << "Offset0"; break;
    case ImmTyOffset1: OS << "Offset1"; break;
    case ImmTyDLC: OS << "DLC"; break;
    case ImmTyGLC: OS << "GLC"; break;
    case ImmTySLC: OS << "SLC"; break;
    case ImmTyTFE: OS << "TFE"; break;
    case ImmTyD16: OS << "D16"; break;
    case ImmTyFORMAT: OS << "FORMAT"; break;
    case ImmTyClampSI: OS << "ClampSI"; break;
    case ImmTyOModSI: OS << "OModSI"; break;
    case ImmTyDPP8: OS << "DPP8"; break;
    case ImmTyDppCtrl: OS << "DppCtrl"; break;
    case ImmTyDppRowMask: OS << "DppRowMask"; break;
    case ImmTyDppBankMask: OS << "DppBankMask"; break;
    case ImmTyDppBoundCtrl: OS << "DppBoundCtrl"; break;
    case ImmTyDppFi: OS << "FI"; break;
    case ImmTySdwaDstSel: OS << "SdwaDstSel"; break;
    case ImmTySdwaSrc0Sel: OS << "SdwaSrc0Sel"; break;
    case ImmTySdwaSrc1Sel: OS << "SdwaSrc1Sel"; break;
    case ImmTySdwaDstUnused: OS << "SdwaDstUnused"; break;
    case ImmTyDMask: OS << "DMask"; break;
    case ImmTyDim: OS << "Dim"; break;
    case ImmTyUNorm: OS << "UNorm"; break;
    case ImmTyDA: OS << "DA"; break;
    case ImmTyR128A16: OS << "R128A16"; break;
    case ImmTyLWE: OS << "LWE"; break;
    case ImmTyOff: OS << "Off"; break;
    case ImmTyExpTgt: OS << "ExpTgt"; break;
    case ImmTyExpCompr: OS << "ExpCompr"; break;
    case ImmTyExpVM: OS << "ExpVM"; break;
    case ImmTyHwreg: OS << "Hwreg"; break;
    case ImmTySendMsg: OS << "SendMsg"; break;
    case ImmTyInterpSlot: OS << "InterpSlot"; break;
    case ImmTyInterpAttr: OS << "InterpAttr"; break;
    case ImmTyAttrChan: OS << "AttrChan"; break;
    case ImmTyOpSel: OS << "OpSel"; break;
    case ImmTyOpSelHi: OS << "OpSelHi"; break;
    case ImmTyNegLo: OS << "NegLo"; break;
    case ImmTyNegHi: OS << "NegHi"; break;
    case ImmTySwizzle: OS << "Swizzle"; break;
    case ImmTyGprIdxMode: OS << "GprIdxMode"; break;
    case ImmTyHigh: OS << "High"; break;
    case ImmTyBLGP: OS << "BLGP"; break;
    case ImmTyCBSZ: OS << "CBSZ"; break;
    case ImmTyABID: OS << "ABID"; break;
    case ImmTyEndpgm: OS << "Endpgm"; break;
    }
  }

  void print(raw_ostream &OS) const override {
    switch (Kind) {
    case KindTy::Register:
      OS << "<register " << getReg() << " mods: " << Reg_ppt.Mods << '>';
      break;
    case KindTy::Immediate:
      OS << '<' << getImmVal();
      if (getImmTy() != ImmTyNone) {
        OS << " type: "; printImmTy(OS, getImmTy());
      }
      OS << " mods: " << Imm.Mods << '>';
      break;
    case KindTy::Token:
      OS << '\'' << getToken() << '\'';
      break;
    case KindTy::Expression:
      OS << "<expr " << *Expr << '>';
      break;
    // riscv
    case KindTy::SystemRegister:
      OS << "<sysreg: " << getSysReg() << '>';
      break;
    }
  }

  static OPUOperand::Ptr CreateImm(const OPUAsmParser *AsmParser,
                                      int64_t Val, SMLoc Loc,
                                      ImmTy Type = ImmTyNone,
                                      bool IsFPImm = false) {
    auto Op = std::make_unique<OPUOperand>(KindTy::Immediate, AsmParser);
    Op->Imm.Val.Int = Val;
    Op->Imm.IsFPImm = IsFPImm;
    Op->Imm.Type = Type;
    Op->Imm.Mods = Modifiers();
    Op->StartLoc = Loc;
    Op->EndLoc = Loc;
    return Op;
  }

  static OPUOperand::Ptr CreateToken(const OPUAsmParser *AsmParser,
                                        StringRef Str, SMLoc Loc,
                                        bool HasExplicitEncodingSize = true) {
    auto Res = std::make_unique<OPUOperand>(KindTy::Token, AsmParser);
    Res->Tok.Data = Str.data();
    Res->Tok.Length = Str.size();
    Res->StartLoc = Loc;
    Res->EndLoc = Loc;
    return Res;
  }

  static OPUOperand::Ptr CreateReg(const OPUAsmParser *AsmParser,
                                      unsigned RegNo, SMLoc S,
                                      SMLoc E) {
    auto Op = std::make_unique<OPUOperand>(KindTy::Register, AsmParser);
    Op->Reg_ppt.RegNo = RegNo;
    Op->Reg_ppt.Mods = Modifiers();
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static OPUOperand::Ptr CreateExpr(const OPUAsmParser *AsmParser,
                                       const class MCExpr *Expr, SMLoc S) {
    auto Op = std::make_unique<OPUOperand>(KindTy::Expression, AsmParser);
    Op->Expr = Expr;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

// below is from RISCV
  // True if operand is a symbol with no modifiers, or a constant with no
  // modifiers and isShiftedInt<N-1, 1>(Op).
  // 
  // 



  static bool evaluateConstantImm(const MCExpr *Expr, int64_t &Imm,
                                  OPUMCExpr::VariantKind &VK) {
    if (auto *RE = dyn_cast<OPUMCExpr>(Expr)) {
      VK = RE->getKind();
      return RE->evaluateAsConstant(Imm);
    }

    if (auto CE = dyn_cast<MCConstantExpr>(Expr)) {
      VK = OPUMCExpr::VK_OPU_None;
      Imm = CE->getValue();
      return true;
    }

    return false;
  }

  template <int N> bool isBareSimmNLsb0() const ;
      /*
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    bool IsValid;
    if (!IsConstantImm)
      IsValid = OPUAsmParser_classifySymbolRef(getImm(), VK, Imm);
    else
      IsValid = isShiftedInt<N - 1, 1>(Imm);
    return IsValid && VK == OPUMCExpr::VK_OPU_None;
  }
  */

  // Predicate methods for AsmOperands defined in OPUInstrInfo.td

  bool isBareSymbol() const ;
      /*
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    // Must be of 'immediate' type but not a constant.
    if (!isImm() || evaluateConstantImm(getImm(), Imm, VK))
      return false;
    return OPUAsmParser_classifySymbolRef(getImm(), VK, Imm) &&
           VK == OPUMCExpr::VK_OPU_None;
  }
  */

  bool isCallSymbol() const ;
      /*
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    // Must be of 'immediate' type but not a constant.
    if (!isImm() || evaluateConstantImm(getImm(), Imm, VK))
      return false;
    return OPUAsmParser_classifySymbolRef(getImm(), VK, Imm) &&
           (VK == OPUMCExpr::VK_OPU_CALL ||
            VK == OPUMCExpr::VK_OPU_CALL_PLT);
  }
  */

  bool isTPRelAddSymbol() const ;
      /*
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    // Must be of 'immediate' type but not a constant.
    if (!isImm() || evaluateConstantImm(getImm(), Imm, VK))
      return false;
    return OPUAsmParser_classifySymbolRef(getImm(), VK, Imm) &&
           VK == OPUMCExpr::VK_OPU_TPREL_ADD;
  }
  */

  bool isCSRSystemRegister() const { return isSystemRegister(); }

  /// Return true if the operand is a valid for the fence instruction e.g.
  /// ('iorw').
  bool isFenceArg() const {
    if (!isImm())
      return false;
    const MCExpr *Val = getImm();
    auto *SVal = dyn_cast<MCSymbolRefExpr>(Val);
    if (!SVal || SVal->getKind() != MCSymbolRefExpr::VK_None)
      return false;

    StringRef Str = SVal->getSymbol().getName();
    // Letters must be unique, taken from 'iorw', and in ascending order. This
    // holds as long as each individual character is one of 'iorw' and is
    // greater than the previous character.
    char Prev = '\0';
    for (char c : Str) {
      if (c != 'i' && c != 'o' && c != 'r' && c != 'w')
        return false;
      if (c <= Prev)
        return false;
      Prev = c;
    }
    return true;
  }

  /// Return true if the operand is a valid floating point rounding mode.
  bool isFRMArg() const {
    if (!isImm())
      return false;
    const MCExpr *Val = getImm();
    auto *SVal = dyn_cast<MCSymbolRefExpr>(Val);
    if (!SVal || SVal->getKind() != MCSymbolRefExpr::VK_None)
      return false;

    StringRef Str = SVal->getSymbol().getName();

    return OPUFPRndMode::stringToRoundingMode(Str) != OPUFPRndMode::Invalid;
  }

  bool isImmXLenLI() const {
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    if (VK == OPUMCExpr::VK_OPU_LO || VK == OPUMCExpr::VK_OPU_PCREL_LO)
      return true;
    // Given only Imm, ensuring that the actually specified constant is either
    // a signed or unsigned 64-bit number is unfortunately impossible.
    bool IsInRange = isRV64() ? true : isInt<32>(Imm) || isUInt<32>(Imm);
    return IsConstantImm && IsInRange && VK == OPUMCExpr::VK_OPU_None;
  }

  // TODO schi copy from rvv
  bool isSImm3() const {
    if (!isImm())
      return false;
    OPUMCExpr::VariantKind VK;
    int64_t Imm;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isInt<3>(Imm) &&
           VK == OPUMCExpr::VK_OPU_None;
  }

  bool isSImm8() const {
    if (!isImm())
      return false;
    OPUMCExpr::VariantKind VK;
    int64_t Imm;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isInt<8>(Imm) &&
           VK == OPUMCExpr::VK_OPU_None;
  }


  bool isUImmLog2XLen() const {
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    if (!isImm())
      return false;
    if (!evaluateConstantImm(getImm(), Imm, VK) ||
        VK != OPUMCExpr::VK_OPU_None)
      return false;
    return (isRV64() && isUInt<6>(Imm)) || isUInt<5>(Imm);
  }

  bool isUImmLog2XLenNonZero() const {
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    if (!isImm())
      return false;
    if (!evaluateConstantImm(getImm(), Imm, VK) ||
        VK != OPUMCExpr::VK_OPU_None)
      return false;
    if (Imm == 0)
      return false;
    return (isRV64() && isUInt<6>(Imm)) || isUInt<5>(Imm);
  }

  bool isUImm5() const {
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isUInt<5>(Imm) && VK == OPUMCExpr::VK_OPU_None;
  }

  bool isUImm5NonZero() const {
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isUInt<5>(Imm) && (Imm != 0) &&
           VK == OPUMCExpr::VK_OPU_None;
  }

  bool isSImm6() const {
    if (!isImm())
      return false;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    int64_t Imm;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isInt<6>(Imm) &&
           VK == OPUMCExpr::VK_OPU_None;
  }

  bool isSImm6NonZero() const {
    if (!isImm())
      return false;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    int64_t Imm;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isInt<6>(Imm) && (Imm != 0) &&
           VK == OPUMCExpr::VK_OPU_None;
  }

  bool isCLUIImm() const {
    if (!isImm())
      return false;
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && (Imm != 0) &&
           (isUInt<5>(Imm) || (Imm >= 0xfffe0 && Imm <= 0xfffff)) &&
           VK == OPUMCExpr::VK_OPU_None;
  }

  bool isUImm7Lsb00() const {
    if (!isImm())
      return false;
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isShiftedUInt<5, 2>(Imm) &&
           VK == OPUMCExpr::VK_OPU_None;
  }


  // TODO schi copy from rvv
  bool isUImm8() const {
    int64_t Imm;
    OPUMCExpr::VariantKind VK;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isUInt<8>(Imm) && VK == OPUMCExpr::VK_OPU_None;
  }

  bool isUImm8Lsb00() const {
    if (!isImm())
      return false;
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isShiftedUInt<6, 2>(Imm) &&
           VK == OPUMCExpr::VK_OPU_None;
  }

  bool isUImm8Lsb000() const {
    if (!isImm())
      return false;
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isShiftedUInt<5, 3>(Imm) &&
           VK == OPUMCExpr::VK_OPU_None;
  }

  bool isSImm9Lsb0() const { return isBareSimmNLsb0<9>(); }

  bool isUImm9Lsb000() const {
    if (!isImm())
      return false;
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isShiftedUInt<6, 3>(Imm) &&
           VK == OPUMCExpr::VK_OPU_None;
  }

  bool isUImm10Lsb00NonZero() const {
    if (!isImm())
      return false;
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && isShiftedUInt<8, 2>(Imm) && (Imm != 0) &&
           VK == OPUMCExpr::VK_OPU_None;
  }

  bool isSImm12() const ;
      /*
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    int64_t Imm;
    bool IsValid;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    if (!IsConstantImm)
      IsValid = OPUAsmParser::classifySymbolRef(getImm(), VK, Imm);
    else
      IsValid = isInt<12>(Imm);
    return IsValid && ((IsConstantImm && VK == OPUMCExpr::VK_OPU_None) ||
                       VK == OPUMCExpr::VK_OPU_LO ||
                       VK == OPUMCExpr::VK_OPU_PCREL_LO ||
                       VK == OPUMCExpr::VK_OPU_TPREL_LO);
  }
  */

  bool isSImm12Lsb0() const { return isBareSimmNLsb0<12>(); }

  bool isSImm13Lsb0() const { return isBareSimmNLsb0<13>(); }

  bool isSImm10Lsb0000NonZero() const ;
  /*
    if (!isImm())
      return false;
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && (Imm != 0) && isShiftedInt<6, 4>(Imm) &&
           VK == OPUMCExpr::VK_OPU_None;
  }
  */

  bool isUImm20LUI() const ;
      /*
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    int64_t Imm;
    bool IsValid;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    if (!IsConstantImm) {
      IsValid = OPUAsmParser_classifySymbolRef(getImm(), VK, Imm);
      return IsValid && (VK == OPUMCExpr::VK_OPU_HI ||
                         VK == OPUMCExpr::VK_OPU_TPREL_HI);
    } else {
      return isUInt<20>(Imm) && (VK == OPUMCExpr::VK_OPU_None ||
                                 VK == OPUMCExpr::VK_OPU_HI ||
                                 VK == OPUMCExpr::VK_OPU_TPREL_HI);
    }
  }
  */

  bool isUImm20AUIPC() const ;
  /*
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    int64_t Imm;
    bool IsValid;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    if (!IsConstantImm) {
      IsValid = OPUAsmParser_classifySymbolRef(getImm(), VK, Imm);
      return IsValid && (VK == OPUMCExpr::VK_OPU_PCREL_HI ||
                         VK == OPUMCExpr::VK_OPU_GOT_HI ||
                         VK == OPUMCExpr::VK_OPU_TLS_GOT_HI ||
                         VK == OPUMCExpr::VK_OPU_TLS_GD_HI);
    } else {
      return isUInt<20>(Imm) && (VK == OPUMCExpr::VK_OPU_None ||
                                 VK == OPUMCExpr::VK_OPU_PCREL_HI ||
                                 VK == OPUMCExpr::VK_OPU_GOT_HI ||
                                 VK == OPUMCExpr::VK_OPU_TLS_GOT_HI ||
                                 VK == OPUMCExpr::VK_OPU_TLS_GD_HI);
    }
  }
  */

  bool isSImm21Lsb0JAL() const ; // { return isBareSimmNLsb0<21>(); }

  bool isImmZero() const ;
      /*
    if (!isImm())
      return false;
    int64_t Imm;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm, VK);
    return IsConstantImm && (Imm == 0) && VK == OPUMCExpr::VK_OPU_None;
  }
  */

  /// getStartLoc - Gets location of the first token of this operand
  SMLoc getStartLoc() const override { return StartLoc; }
  /// getEndLoc - Gets location of the last token of this operand
  SMLoc getEndLoc() const override { return EndLoc; }
  /// True if this operand is for an RV64 instruction
  bool isRV64() const { return IsRV64; }
/*
  unsigned getReg() const override {
    assert(Kind == KindTy::Register && "Invalid type access!");
    return Reg.RegNum.id();
  }
  */

  StringRef getSysReg() const {
    assert(Kind == KindTy::SystemRegister && "Invalid access!");
    return StringRef(SysReg.Data, SysReg.Length);
  }

  const MCExpr *getImm() const {
    assert(Kind == KindTy::Immediate && "Invalid type access!");
    return Imm.Val.MCExpr;
  }

  const int64_t getImmInt() const {
    assert(Kind == KindTy::Immediate && "Invalid type access!");
    return Imm.Val.Int;
  }

/* TODO keep AMD print
  StringRef getToken_base() const {
    assert(Kind == KindTy::Token && "Invalid type access!");
    return Tok;
  }
  void print(raw_ostream &OS) const override {
    switch (Kind) {
    case KindTy::Immediate:
      OS << *getImm();
      break;
    case KindTy::Register:
      OS << "<register x";
      OS << getReg() << ">";
      break;
    case KindTy::Token:
      OS << "'" << getToken_base() << "'";
      break;
    case KindTy::SystemRegister:
      OS << "<sysreg: " << getSysReg() << '>';
      break;
    }
  }
*/
  static std::unique_ptr<OPUOperand> createToken(StringRef Str, SMLoc S,
                                                   bool IsRV64) {
    auto Op = std::make_unique<OPUOperand>(KindTy::Token);
    Op->Tok_StringRef = Str;
    Op->StartLoc = S;
    Op->EndLoc = S;
    Op->IsRV64 = IsRV64;
    return Op;
  }

  static std::unique_ptr<OPUOperand> createReg(unsigned RegNo, SMLoc S,
                                                 SMLoc E, bool IsRV64) {
    auto Op = std::make_unique<OPUOperand>(KindTy::Register);
    Op->Reg.RegNum = RegNo;
    Op->StartLoc = S;
    Op->EndLoc = E;
    Op->IsRV64 = IsRV64;
    return Op;
  }

  static std::unique_ptr<OPUOperand> createImm(const MCExpr *Val, SMLoc S,
                                                 SMLoc E, bool IsRV64) {
    auto Op = std::make_unique<OPUOperand>(KindTy::Immediate);
    Op->Imm.Val.MCExpr = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    Op->IsRV64 = IsRV64;
    return Op;
  }

  static std::unique_ptr<OPUOperand>
  createSysReg(StringRef Str, SMLoc S, unsigned Encoding, bool IsRV64) {
    auto Op = std::make_unique<OPUOperand>(KindTy::SystemRegister);
    Op->SysReg.Data = Str.data();
    Op->SysReg.Length = Str.size();
    Op->SysReg.Encoding = Encoding;
    Op->StartLoc = S;
    Op->IsRV64 = IsRV64;
    return Op;
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    assert(Expr && "Expr shouldn't be null!");
    int64_t Imm = 0;
    OPUMCExpr::VariantKind VK = OPUMCExpr::VK_OPU_None;
    bool IsConstant = evaluateConstantImm(Expr, Imm, VK);

    if (IsConstant)
      Inst.addOperand(MCOperand::createImm(Imm));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }
/* TODO remove riscv
  // Used by the TableGen Code
  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }
*/
  void addFenceArgOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    // isFenceArg has validated the operand, meaning this cast is safe
    auto SE = cast<MCSymbolRefExpr>(getImm());

    unsigned Imm = 0;
    for (char c : SE->getSymbol().getName()) {
      switch (c) {
      default:
        llvm_unreachable("FenceArg must contain only [iorw]");
      case 'i': Imm |= OPUFenceField::I; break;
      case 'o': Imm |= OPUFenceField::O; break;
      case 'r': Imm |= OPUFenceField::R; break;
      case 'w': Imm |= OPUFenceField::W; break;
      }
    }
    Inst.addOperand(MCOperand::createImm(Imm));
  }

  void addCSRSystemRegisterOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createImm(SysReg.Encoding));
  }

  // Returns the rounding mode represented by this OPUOperand. Should only
  // be called after checking isFRMArg.
  OPUFPRndMode::RoundingMode getRoundingMode() const {
    // isFRMArg has validated the operand, meaning this cast is safe.
    auto SE = cast<MCSymbolRefExpr>(getImm());
    OPUFPRndMode::RoundingMode FRM =
        OPUFPRndMode::stringToRoundingMode(SE->getSymbol().getName());
    assert(FRM != OPUFPRndMode::Invalid && "Invalid rounding mode");
    return FRM;
  }

  void addFRMArgOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createImm(getRoundingMode()));
  }


};

class OPUAsmParser : public MCTargetAsmParser {
  SmallVector<FeatureBitset, 4> FeatureBitStack;

/* AMD */
  MCAsmParser &Parser;
  // Number of extra operands parsed after the first optional operand.
  // This may be necessary to skip hardcoded mandatory operands.
  static const unsigned MAX_OPR_LOOKAHEAD = 8;

  unsigned ForcedEncodingSize = 0;
  bool ForcedDPP = false;
  bool ForcedSDWA = false;
  KernelScopeInfo KernelScope;
/* AMD */


  // SMLoc getLoc() const { return getParser().getTok().getLoc(); }
  bool isRV64() const { return getSTI().hasFeature(OPU::Feature64Bit); }
  bool isRV32E() const { return getSTI().hasFeature(OPU::FeatureRV32E); }
  bool isPPT() const { return getSTI().hasFeature(OPU::FeaturePPT); }

  OPUTargetStreamer &getTargetStreamer() {
    MCTargetStreamer &TS = *getParser().getStreamer().getTargetStreamer();
    return static_cast<OPUTargetStreamer &>(TS);
  }

  unsigned validateTargetOperandClass(MCParsedAsmOperand &Op,
                                      unsigned Kind) override;

  bool generateImmOutOfRangeError(OperandVector &Operands, uint64_t ErrorInfo,
                                  int64_t Lower, int64_t Upper, Twine Msg);

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;

  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  bool ParseDirective(AsmToken DirectiveID) override;

  // Helper to actually emit an instruction to the MCStreamer. Also, when
  // possible, compression of the instruction is performed.
  void emitToStreamer(MCStreamer &S, const MCInst &Inst);

  // Helper to emit a combination of LUI, ADDI(W), and SLLI instructions that
  // synthesize the desired immedate value into the destination register.
  void emitLoadImm(Register DestReg, int64_t Value, MCStreamer &Out);

  // Helper to emit a combination of AUIPC and SecondOpcode. Used to implement
  // helpers such as emitLoadLocalAddress and emitLoadAddress.
  void emitAuipcInstPair(MCOperand DestReg, MCOperand TmpReg,
                         const MCExpr *Symbol, OPUMCExpr::VariantKind VKHi,
                         unsigned SecondOpcode, SMLoc IDLoc, MCStreamer &Out);

  // Helper to emit pseudo instruction "lla" used in PC-rel addressing.
  void emitLoadLocalAddress(MCInst &Inst, SMLoc IDLoc, MCStreamer &Out);

  // Helper to emit pseudo instruction "la" used in GOT/PC-rel addressing.
  void emitLoadAddress(MCInst &Inst, SMLoc IDLoc, MCStreamer &Out);

  // Helper to emit pseudo instruction "la.tls.ie" used in initial-exec TLS
  // addressing.
  void emitLoadTLSIEAddress(MCInst &Inst, SMLoc IDLoc, MCStreamer &Out);

  // Helper to emit pseudo instruction "la.tls.gd" used in global-dynamic TLS
  // addressing.
  void emitLoadTLSGDAddress(MCInst &Inst, SMLoc IDLoc, MCStreamer &Out);

  // Helper to emit pseudo load/store instruction with a symbol.
  void emitLoadStoreSymbol(MCInst &Inst, unsigned Opcode, SMLoc IDLoc,
                           MCStreamer &Out, bool HasTmpReg);

  // Checks that a PseudoAddTPRel is using x4/tp in its second input operand.
  // Enforcing this using a restricted register class for the second input
  // operand of PseudoAddTPRel results in a poor diagnostic due to the fact
  // 'add' is an overloaded mnemonic.
  bool checkPseudoAddTPRel(MCInst &Inst, OperandVector &Operands);

  /// Helper for processing MC instructions that have been successfully matched
  /// by MatchAndEmitInstruction. Modifications to the emitted instructions,
  /// like the expansion of pseudo instructions (e.g., "li"), can be performed
  /// in this method.
  bool processInstruction(MCInst &Inst, SMLoc IDLoc, OperandVector &Operands,
                          MCStreamer &Out);

// Auto-generated instruction matching functions
#define GET_ASSEMBLER_HEADER
#include "OPUGenAsmMatcher.inc"
  
  /*AMD*/
private:
  bool ParseAsAbsoluteExpression(uint32_t &Ret);
  bool OutOfRangeError(SMRange Range);
  /// Calculate VGPR/SGPR blocks required for given target, reserved
  /// registers, and user-specified NextFreeXGPR values.
  ///
  /// \param Features [in] Target features, used for bug corrections.
  /// \param VCCUsed [in] Whether VCC special SGPR is reserved.
  /// \param FlatScrUsed [in] Whether FLAT_SCRATCH special SGPR is reserved.
  /// \param XNACKUsed [in] Whether XNACK_MASK special SGPR is reserved.
  /// \param EnableWavefrontSize32 [in] Value of ENABLE_WAVEFRONT_SIZE32 kernel
  /// descriptor field, if valid.
  /// \param NextFreeVGPR [in] Max VGPR number referenced, plus one.
  /// \param VGPRRange [in] Token range, used for VGPR diagnostics.
  /// \param NextFreeSGPR [in] Max SGPR number referenced, plus one.
  /// \param SGPRRange [in] Token range, used for SGPR diagnostics.
  /// \param VGPRBlocks [out] Result VGPR block count.
  /// \param SGPRBlocks [out] Result SGPR block count.
  bool calculateGPRBlocks(const FeatureBitset &Features, bool VCCUsed,
                          bool FlatScrUsed, bool XNACKUsed,
                          Optional<bool> EnableWavefrontSize32, unsigned NextFreeVGPR,
                          SMRange VGPRRange, unsigned NextFreeSGPR,
                          SMRange SGPRRange, unsigned &VGPRBlocks,
                          unsigned &SGPRBlocks);
  bool ParseDirectiveOPUTarget();
  bool ParseDirectivePPSKernel();
  bool ParseDirectiveMajorMinor(uint32_t &Major, uint32_t &Minor);
  // bool ParseDirectiveHSACodeObjectVersion();
  // bool ParseDirectiveHSACodeObjectISA();
  bool ParseAMDKernelCodeTValue(StringRef ID, amd_kernel_code_t &Header);
  // bool ParseDirectiveAMDKernelCodeT();
  // bool subtargetHasRegister(const MCRegisterInfo &MRI, unsigned RegNo) const;
  // bool ParseDirectiveOPUHsaKernel();

  // bool ParseDirectiveISAVersion();
  bool ParseDirectivePPSMetadata();
  // bool ParseDirectivePALMetadataBegin();
  // bool ParseDirectivePALMetadata();
  bool ParseDirectiveOPULDS();

  /// Common code to parse out a block of text (typically YAML) between start and
  /// end directives.
  bool ParseToEndDirective(const char *AssemblerDirectiveBegin,
                           const char *AssemblerDirectiveEnd,
                           std::string &CollectString);

  bool AddNextRegisterToList(unsigned& Reg, unsigned& RegWidth,
                             RegisterKind RegKind, unsigned Reg1,
                             unsigned RegNum);
  bool ParseOPURegister(RegisterKind& RegKind, unsigned& Reg,
                           unsigned& RegNum, unsigned& RegWidth,
                           unsigned *DwordRegIndex);
  bool isRegister();
  bool isRegister(const AsmToken &Token, const AsmToken &NextToken) const;
  Optional<StringRef> getGprCountSymbolName(RegisterKind RegKind);
  void initializeGprCountSymbol(RegisterKind RegKind);
  bool updateGprCountSymbols(RegisterKind RegKind, unsigned DwordRegIndex,
                             unsigned RegWidth);
  void cvtMubufImpl(MCInst &Inst, const OperandVector &Operands,
                    bool IsAtomic, bool IsAtomicReturn, bool IsLds = false);
  void cvtDSImpl(MCInst &Inst, const OperandVector &Operands,
                 bool IsGdsHardcoded);

public:
  /* use riscv one
  enum OPUMatchResultTy {
    Match_PreferE32 = FIRST_TARGET_MATCH_RESULT_TY
  };
  */
  enum OperandMode {
    OperandMode_Default,
    OperandMode_NSA,
  };

  using OptionalImmIndexMap = std::map<OPUOperand::ImmTy, unsigned>;

  /*end AMD*/





  OperandMatchResultTy parseCSRSystemRegister(OperandVector &Operands);
  OperandMatchResultTy parseImmediate(OperandVector &Operands);
  OperandMatchResultTy parseRegister(OperandVector &Operands,
                                     bool AllowParens = false);
  OperandMatchResultTy parseMemOpBaseReg(OperandVector &Operands);
  OperandMatchResultTy parseAtomicMemOp(OperandVector &Operands);
  OperandMatchResultTy parseOperandWithModifier(OperandVector &Operands);
  OperandMatchResultTy parseBareSymbol(OperandVector &Operands);
  OperandMatchResultTy parseCallSymbol(OperandVector &Operands);
  OperandMatchResultTy parseJALOffset(OperandVector &Operands);

  bool parseOperand(OperandVector &Operands, StringRef Mnemonic);

  bool parseDirectiveOption();

  void setFeatureBits(uint64_t Feature, StringRef FeatureString) {
    if (!(getSTI().getFeatureBits()[Feature])) {
      MCSubtargetInfo &STI = copySTI();
      setAvailableFeatures(
          ComputeAvailableFeatures(STI.ToggleFeature(FeatureString)));
    }
  }

  void clearFeatureBits(uint64_t Feature, StringRef FeatureString) {
    if (getSTI().getFeatureBits()[Feature]) {
      MCSubtargetInfo &STI = copySTI();
      setAvailableFeatures(
          ComputeAvailableFeatures(STI.ToggleFeature(FeatureString)));
    }
  }

  void pushFeatureBits() {
    FeatureBitStack.push_back(getSTI().getFeatureBits());
  }

  bool popFeatureBits() {
    if (FeatureBitStack.empty())
      return true;

    FeatureBitset FeatureBits = FeatureBitStack.pop_back_val();
    copySTI().setFeatureBits(FeatureBits);
    setAvailableFeatures(ComputeAvailableFeatures(FeatureBits));

    return false;
  }
public:
  enum OPUMatchResultTy {
    Match_PreferE32 = FIRST_TARGET_MATCH_RESULT_TY,
#define GET_OPERAND_DIAGNOSTIC_TYPES
#include "OPUGenAsmMatcher.inc"
#undef GET_OPERAND_DIAGNOSTIC_TYPES
  };

  static bool classifySymbolRef(const MCExpr *Expr,
                                OPUMCExpr::VariantKind &Kind,
                                int64_t &Addend);

  OPUAsmParser(const MCSubtargetInfo &STI, MCAsmParser &_Parser,
                 const MCInstrInfo &MII, const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, STI, MII), Parser(_Parser) {
    MCAsmParserExtension::Initialize(Parser);

    Parser.addAliasForDirective(".half", ".2byte");
    Parser.addAliasForDirective(".hword", ".2byte");
    Parser.addAliasForDirective(".word", ".4byte");
    Parser.addAliasForDirective(".dword", ".8byte");
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));

    {
      // TODO: make those pre-defined variables read-only.
      // Currently there is none suitable machinery in the core llvm-mc for this.
      // MCSymbol::isRedefinable is intended for another purpose, and
      // AsmParser::parseDirectiveSet() cannot be specialized for specific target.
      OPU::IsaVersion ISA = {0, 0, 0}; // OPU::getIsaVersion(getSTI().getCPU());
      MCContext &Ctx = getContext();
      if (OPU::IsaInfo::hasCodeObjectV3(&getSTI())) {
          /*
        MCSymbol *Sym =
            Ctx.getOrCreateSymbol(Twine(".amdgcn.gfx_generation_number"));
        Sym->setVariableValue(MCConstantExpr::create(ISA.Major, Ctx));
        Sym = Ctx.getOrCreateSymbol(Twine(".amdgcn.gfx_generation_minor"));
        Sym->setVariableValue(MCConstantExpr::create(ISA.Minor, Ctx));
        Sym = Ctx.getOrCreateSymbol(Twine(".amdgcn.gfx_generation_stepping"));
        Sym->setVariableValue(MCConstantExpr::create(ISA.Stepping, Ctx));
        */
      }

      if (OPU::IsaInfo::hasCodeObjectV3(&getSTI())) {
          // FIXME add more regkind
        initializeGprCountSymbol(IS_VGPR);
        initializeGprCountSymbol(IS_SGPR);
      }
    }

  }

  // below AMD
  bool hasXNACK() const {
    return false; // OPU::hasXNACK(getSTI());
  }

  bool hasMIMG_R128() const {
    return false; // OPU::hasMIMG_R128(getSTI());
  }

  bool hasPackedD16() const {
    return true; // OPU::hasPackedD16(getSTI());
  }

  bool isSI() const {
    return false; // OPU::isSI(getSTI());
  }

  bool isCI() const {
    return false; // OPU::isCI(getSTI());
  }

  bool isVI() const {
    return true; // OPU::isVI(getSTI());
  }

  bool isGFX9() const {
    return true; // OPU::isGFX9(getSTI());
  }

  bool isGFX10() const {
    return true; // OPU::isGFX10(getSTI());
  }

  bool hasInv2PiInlineImm() const {
    return getFeatureBits()[OPU::FeatureInv2PiInlineImm];
  }

  bool hasFlatOffsets() const {
    return getFeatureBits()[OPU::FeatureFlatInstOffsets];
  }

  bool hasSGPR102_SGPR103() const {
    return !isVI() && !isGFX9();
  }

  bool hasSGPR104_SGPR105() const {
    return isGFX10();
  }

  bool hasIntClamp() const {
    return getFeatureBits()[OPU::FeatureIntClamp];
  }
/*
  OPUTargetStreamer &getTargetStreamer() {
    MCTargetStreamer &TS = *getParser().getStreamer().getTargetStreamer();
    return static_cast<OPUTargetStreamer &>(TS);
  }
*/
  const MCRegisterInfo *getMRI() const {
    // We need this const_cast because for some reason getContext() is not const
    // in MCAsmParser.
    return const_cast<OPUAsmParser*>(this)->getContext().getRegisterInfo();
  }

  const MCInstrInfo *getMII() const {
    return &MII;
  }

  const FeatureBitset &getFeatureBits() const {
    return getSTI().getFeatureBits();
  }

  void setForcedEncodingSize(unsigned Size) { ForcedEncodingSize = Size; }
  void setForcedDPP(bool ForceDPP_) { ForcedDPP = ForceDPP_; }
  void setForcedSDWA(bool ForceSDWA_) { ForcedSDWA = ForceSDWA_; }

  unsigned getForcedEncodingSize() const { return ForcedEncodingSize; }
  bool isForcedVOP3() const { return ForcedEncodingSize == 64; }
  bool isForcedDPP() const { return ForcedDPP; }
  bool isForcedSDWA() const { return ForcedSDWA; }
  ArrayRef<unsigned> getMatchedVariants() const;

  std::unique_ptr<OPUOperand> parseRegister();

  // bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;
  // 
  unsigned checkTargetMatchPredicate(MCInst &Inst) override;
  //unsigned validateTargetOperandClass(MCParsedAsmOperand &Op,
  //                                    unsigned Kind) override;
  bool MatchAndEmitInstruction_ppt(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                                uint64_t &ErrorInfo,
                                bool MatchingInlineAsm) ;
  // bool ParseDirective(AsmToken DirectiveID) override;
  OperandMatchResultTy parseOperand_ppt(OperandVector &Operands, StringRef Mnemonic,
                                    OperandMode Mode = OperandMode_Default);
  StringRef parseMnemonicSuffix(StringRef Name);
  bool ParseInstruction_ppt(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) ;
  // bool ProcessInstruction(MCInst &Inst);

  OperandMatchResultTy parseIntWithPrefix(const char *Prefix, int64_t &Int);

  OperandMatchResultTy
  parseIntWithPrefix(const char *Prefix, OperandVector &Operands,
                     OPUOperand::ImmTy ImmTy = OPUOperand::ImmTyNone,
                     bool (*ConvertResult)(int64_t &) = nullptr);

  OperandMatchResultTy
  parseOperandArrayWithPrefix(const char *Prefix,
                              OperandVector &Operands,
                              OPUOperand::ImmTy ImmTy = OPUOperand::ImmTyNone,
                              bool (*ConvertResult)(int64_t&) = nullptr);

  OperandMatchResultTy
  parseNamedBit(const char *Name, OperandVector &Operands,
                OPUOperand::ImmTy ImmTy = OPUOperand::ImmTyNone);
  OperandMatchResultTy parseStringWithPrefix(StringRef Prefix,
                                             StringRef &Value);

  bool isModifier();
  bool isOperandModifier(const AsmToken &Token, const AsmToken &NextToken) const;
  bool isRegOrOperandModifier(const AsmToken &Token, const AsmToken &NextToken) const;
  bool isNamedOperandModifier(const AsmToken &Token, const AsmToken &NextToken) const;
  bool isOpcodeModifierWithVal(const AsmToken &Token, const AsmToken &NextToken) const;
  bool parseSP3NegModifier();
  OperandMatchResultTy parseImm(OperandVector &Operands, bool HasSP3AbsModifier = false);
  OperandMatchResultTy parseReg(OperandVector &Operands);
  OperandMatchResultTy parseRegOrImm(OperandVector &Operands, bool HasSP3AbsMod = false);
  OperandMatchResultTy parseRegOrImmWithFPInputMods(OperandVector &Operands, bool AllowImm = true);
  OperandMatchResultTy parseRegOrImmWithIntInputMods(OperandVector &Operands, bool AllowImm = true);
  OperandMatchResultTy parseRegWithFPInputMods(OperandVector &Operands);
  OperandMatchResultTy parseRegWithIntInputMods(OperandVector &Operands);
  OperandMatchResultTy parseVReg32OrOff(OperandVector &Operands);
  OperandMatchResultTy parseDfmtNfmt(OperandVector &Operands);

  void cvtDSOffset01(MCInst &Inst, const OperandVector &Operands);
  void cvtDS(MCInst &Inst, const OperandVector &Operands) { cvtDSImpl(Inst, Operands, false); }
  void cvtDSGds(MCInst &Inst, const OperandVector &Operands) { cvtDSImpl(Inst, Operands, true); }
  void cvtExp(MCInst &Inst, const OperandVector &Operands);

  bool parseCnt(int64_t &IntVal);
  OperandMatchResultTy parseSWaitCntOps(OperandVector &Operands);
  OperandMatchResultTy parseHwreg(OperandVector &Operands) {
    return MatchOperand_Success;
  }

private:
  struct OperandInfoTy {
    int64_t Id;
    bool IsSymbolic = false;
    bool IsDefined = false;

    OperandInfoTy(int64_t Id_) : Id(Id_) {}
  };

  bool parseSendMsgBody(OperandInfoTy &Msg, OperandInfoTy &Op, OperandInfoTy &Stream);
  bool validateSendMsg(const OperandInfoTy &Msg,
                       const OperandInfoTy &Op,
                       const OperandInfoTy &Stream,
                       const SMLoc Loc);

  bool parseHwregBody(OperandInfoTy &HwReg, int64_t &Offset, int64_t &Width);
  bool validateHwreg(const OperandInfoTy &HwReg,
                     const int64_t Offset,
                     const int64_t Width,
                     const SMLoc Loc);

  void errorExpTgt();
  OperandMatchResultTy parseExpTgtImpl(StringRef Str, uint8_t &Val);
  SMLoc getFlatOffsetLoc(const OperandVector &Operands) const;

  bool validateInstruction(const MCInst &Inst, const SMLoc &IDLoc, const OperandVector &Operands);
  bool validateFlatOffset(const MCInst &Inst, const OperandVector &Operands);
  bool validateSOPLiteral(const MCInst &Inst) const;
  bool validateConstantBusLimitations(const MCInst &Inst);
  bool validateEarlyClobberLimitations(const MCInst &Inst);
  bool validateIntClampSupported(const MCInst &Inst);
  bool validateMIMGAtomicDMask(const MCInst &Inst);
  bool validateMIMGGatherDMask(const MCInst &Inst);
  bool validateMIMGDataSize(const MCInst &Inst);
  bool validateMIMGAddrSize(const MCInst &Inst);
  bool validateMIMGD16(const MCInst &Inst);
  bool validateMIMGDim(const MCInst &Inst);
  bool validateLdsDirect(const MCInst &Inst);
  bool validateOpSel(const MCInst &Inst);
  bool validateVccOperand(unsigned Reg) const;
  bool validateVOP3Literal(const MCInst &Inst) const;
  bool usesConstantBus(const MCInst &Inst, unsigned OpIdx);
  bool isInlineConstant(const MCInst &Inst, unsigned OpIdx) const;
  unsigned findImplicitSGPRReadInVOP(const MCInst &Inst) const;

  bool isId(const StringRef Id) const;
  bool isId(const AsmToken &Token, const StringRef Id) const;
  bool isToken(const AsmToken::TokenKind Kind) const;
  bool trySkipId(const StringRef Id);
  bool trySkipId(const StringRef Id, const AsmToken::TokenKind Kind);
  bool trySkipToken(const AsmToken::TokenKind Kind);
  bool skipToken(const AsmToken::TokenKind Kind, const StringRef ErrMsg);
  bool parseString(StringRef &Val, const StringRef ErrMsg = "expected a string");
  void peekTokens(MutableArrayRef<AsmToken> Tokens);
  AsmToken::TokenKind getTokenKind() const;
  bool parseExpr(int64_t &Imm);
  bool parseExpr(OperandVector &Operands);
  StringRef getTokenStr() const;
  AsmToken peekToken();
  AsmToken getToken() const;
  SMLoc getLoc() const;
  void lex();

public:
  OperandMatchResultTy parseOptionalOperand(OperandVector &Operands);
  OperandMatchResultTy parseOptionalOpr(OperandVector &Operands);

  OperandMatchResultTy parseExpTgt(OperandVector &Operands);
  OperandMatchResultTy parseSendMsgOp(OperandVector &Operands);
  OperandMatchResultTy parseInterpSlot(OperandVector &Operands);
  OperandMatchResultTy parseInterpAttr(OperandVector &Operands);
  OperandMatchResultTy parseSOppBrTarget(OperandVector &Operands);
  OperandMatchResultTy parseBoolReg(OperandVector &Operands);

  bool parseSwizzleOperands(const unsigned OpNum, int64_t* Op,
                            const unsigned MinVal,
                            const unsigned MaxVal,
                            const StringRef ErrMsg);
  OperandMatchResultTy parseSwizzleOp(OperandVector &Operands);
  bool parseSwizzleOffset(int64_t &Imm);
  bool parseSwizzleMacro(int64_t &Imm);
  bool parseSwizzleQuadPerm(int64_t &Imm);
  bool parseSwizzleBitmaskPerm(int64_t &Imm);
  bool parseSwizzleBroadcast(int64_t &Imm);
  bool parseSwizzleSwap(int64_t &Imm);
  bool parseSwizzleReverse(int64_t &Imm);

  OperandMatchResultTy parseVPRIdxMode(OperandVector &Operands);
  int64_t parseGPRIdxMacro();

  void cvtMubuf(MCInst &Inst, const OperandVector &Operands) { cvtMubufImpl(Inst, Operands, false, false); }
  void cvtMubufAtomic(MCInst &Inst, const OperandVector &Operands) { cvtMubufImpl(Inst, Operands, true, false); }
  void cvtMubufAtomicReturn(MCInst &Inst, const OperandVector &Operands) { cvtMubufImpl(Inst, Operands, true, true); }
  void cvtMubufLds(MCInst &Inst, const OperandVector &Operands) { cvtMubufImpl(Inst, Operands, false, false, true); }
  void cvtMtbuf(MCInst &Inst, const OperandVector &Operands);

  OPUOperand::Ptr defaultDLC() const;
  OPUOperand::Ptr defaultGLC() const;
  OPUOperand::Ptr defaultSLC() const;

  OPUOperand::Ptr defaultSMRDOffset8() const;
  OPUOperand::Ptr defaultSMRDOffset20() const;
  OPUOperand::Ptr defaultSMRDLiteralOffset() const;
  OPUOperand::Ptr defaultFlatOffset() const;

  OperandMatchResultTy parseOModOperand(OperandVector &Operands);

  void cvtVOP3(MCInst &Inst, const OperandVector &Operands,
               OptionalImmIndexMap &OptionalIdx);
  void cvtVOP3OpSel(MCInst &Inst, const OperandVector &Operands);
  void cvtVOP3(MCInst &Inst, const OperandVector &Operands);
  void cvtVOP3P(MCInst &Inst, const OperandVector &Operands);

  // void cvtVOP3Interp(MCInst &Inst, const OperandVector &Operands);

  // void cvtMIMG(MCInst &Inst, const OperandVector &Operands,
  //             bool IsAtomic = false);
  // void cvtMIMGAtomic(MCInst &Inst, const OperandVector &Operands);

  // OperandMatchResultTy parseDim(OperandVector &Operands);
  // OperandMatchResultTy parseDPP8(OperandVector &Operands);
  // OperandMatchResultTy parseDPPCtrl(OperandVector &Operands);
  OPUOperand::Ptr defaultRowMask() const;
  OPUOperand::Ptr defaultBankMask() const;
  OPUOperand::Ptr defaultBoundCtrl() const;
  OPUOperand::Ptr defaultFI() const;
  // void cvtDPP(MCInst &Inst, const OperandVector &Operands, bool IsDPP8 = false);
  // void cvtDPP8(MCInst &Inst, const OperandVector &Operands) { cvtDPP(Inst, Operands, true); }

  // OperandMatchResultTy parseSDWASel(OperandVector &Operands, StringRef Prefix,
  //                                   OPUOperand::ImmTy Type);
  // OperandMatchResultTy parseSDWADstUnused(OperandVector &Operands);
  // void cvtSdwaVOP1(MCInst &Inst, const OperandVector &Operands);
  // void cvtSdwaVOP2(MCInst &Inst, const OperandVector &Operands);
  // void cvtSdwaVOP2b(MCInst &Inst, const OperandVector &Operands);
  // void cvtSdwaVOPC(MCInst &Inst, const OperandVector &Operands);
  // void cvtSDWA(MCInst &Inst, const OperandVector &Operands,
  //               uint64_t BasicInstType, bool skipVcc = false);

  OPUOperand::Ptr defaultBLGP() const;
  OPUOperand::Ptr defaultCBSZ() const;
  OPUOperand::Ptr defaultABID() const;

  OperandMatchResultTy parseEndpgmOp(OperandVector &Operands);
  OPUOperand::Ptr defaultEndpgmImmOperands() const;


};

//===----------------------------------------------------------------------===//
// Operand
//===----------------------------------------------------------------------===//


raw_ostream &operator <<(raw_ostream &OS, OPUOperand::Modifiers Mods) {
  OS << "abs:" << Mods.Abs << " neg: " << Mods.Neg << " sext:" << Mods.Sext;
  return OS;
}


struct OptionalOperand {
  const char *Name;
  OPUOperand::ImmTy Type;
  bool IsBit;
  bool (*ConvertResult)(int64_t&);
};

} // end anonymous namespace.


