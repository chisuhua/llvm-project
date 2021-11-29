//===-- OPUMCInstLower.cpp - Convert OPU MachineInstr to an MCInst ------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower OPU MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "OPU.h"
#include "OPUInstrInfo.h"
#include "OPUMCInstLower.h"
#include "MCTargetDesc/OPUMCExpr.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// FIXME OPU need to merge Flag Kind with AMD getVariantKind
MCOperand OPUMCInstLower::lowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym) const {
  MCContext &Ctx = AP.OutContext;
  OPUMCExpr::VariantKind Kind;

  switch (MO.getTargetFlags()) {
  default:
    // llvm_unreachable("Unknown target flag on GV operand");
    // FIXME I merge AMD here
    return MCOperand::createExpr(
                getLongBranchBlockExpr(*MO.getParent()->getParent(), MO));
  case OPUII::MO_None:
    Kind = OPUMCExpr::VK_OPU_None;
    break;
  case OPUII::MO_CALL:
    Kind = OPUMCExpr::VK_OPU_CALL;
    break;
  case OPUII::MO_PLT:
    Kind = OPUMCExpr::VK_OPU_CALL_PLT;
    break;
  case OPUII::MO_LO:
    Kind = OPUMCExpr::VK_OPU_LO;
    break;
  case OPUII::MO_HI:
    Kind = OPUMCExpr::VK_OPU_HI;
    break;
  case OPUII::MO_PCREL_LO:
    Kind = OPUMCExpr::VK_OPU_PCREL_LO;
    break;
  case OPUII::MO_PCREL_HI:
    Kind = OPUMCExpr::VK_OPU_PCREL_HI;
    break;
  case OPUII::MO_GOT_HI:
    Kind = OPUMCExpr::VK_OPU_GOT_HI;
    break;
  case OPUII::MO_TPREL_LO:
    Kind = OPUMCExpr::VK_OPU_TPREL_LO;
    break;
  case OPUII::MO_TPREL_HI:
    Kind = OPUMCExpr::VK_OPU_TPREL_HI;
    break;
  case OPUII::MO_TPREL_ADD:
    Kind = OPUMCExpr::VK_OPU_TPREL_ADD;
    break;
  case OPUII::MO_TLS_GOT_HI:
    Kind = OPUMCExpr::VK_OPU_TLS_GOT_HI;
    break;
  case OPUII::MO_TLS_GD_HI:
    Kind = OPUMCExpr::VK_OPU_TLS_GD_HI;
    break;
  }

  const MCExpr *ME =
      MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None, Ctx);

  if (!MO.isJTI() && !MO.isMBB() && MO.getOffset())
    ME = MCBinaryExpr::createAdd(
        ME, MCConstantExpr::create(MO.getOffset(), Ctx), Ctx);

  if (Kind != OPUMCExpr::VK_OPU_None)
    ME = OPUMCExpr::create(ME, Kind, Ctx);
  return MCOperand::createExpr(ME);
}

// from AMD
OPUMCInstLower::OPUMCInstLower(MCContext &ctx,
                                     const TargetSubtargetInfo &st,
                                     const AsmPrinter &ap):
  Ctx(ctx), ST(st), AP(ap) { }

static MCSymbolRefExpr::VariantKind getVariantKind(unsigned MOFlags) {
  switch (MOFlags) {
  default:
    return MCSymbolRefExpr::VK_None;
  case OPUInstrInfo::MO_GOTPCREL:
    return MCSymbolRefExpr::VK_GOTPCREL;
  case OPUInstrInfo::MO_GOTPCREL32_LO:
    return MCSymbolRefExpr::VK_OPU_GOTPCREL32_LO;
  case OPUInstrInfo::MO_GOTPCREL32_HI:
    return MCSymbolRefExpr::VK_OPU_GOTPCREL32_HI;
  case OPUInstrInfo::MO_REL32_LO:
    return MCSymbolRefExpr::VK_OPU_REL32_LO;
  case OPUInstrInfo::MO_REL32_HI:
    return MCSymbolRefExpr::VK_OPU_REL32_HI;
  case OPUInstrInfo::MO_PCREL32_LO:
    return MCSymbolRefExpr::VK_OPU_PCREL32_LO;
  case OPUInstrInfo::MO_PCREL32_HI:
    return MCSymbolRefExpr::VK_OPU_PCREL32_HI;
  case OPUInstrInfo::MO_PCREL_CALL:
    return MCSymbolRefExpr::VK_OPU_PCREL_CALL;
  case OPUInstrInfo::MO_ABS32_LO:
    return MCSymbolRefExpr::VK_OPU_ABS32_LO;
  case OPUInstrInfo::MO_ABS32_HI:
    return MCSymbolRefExpr::VK_OPU_ABS32_HI;
  }
}

// from AMD
const MCExpr *OPUMCInstLower::getLongBranchBlockExpr(
  const MachineBasicBlock &SrcBB,
  const MachineOperand &MO) const {
  const MCExpr *DestBBSym = MCSymbolRefExpr::create(MO.getMBB()->getSymbol(), Ctx);
  const MCExpr *SrcBBSym = MCSymbolRefExpr::create(SrcBB.getSymbol(), Ctx);

  // FIXME: The first half of this assert should be removed. This should
  // probably be PC relative instead of using the source block symbol, and
  // therefore the indirect branch expansion should use a bundle.
  assert(
      skipDebugInstructionsForward(SrcBB.begin(), SrcBB.end())->getOpcode() ==
          OPU::S_GETPC_B64 &&
      ST.getInstrInfo()->get(OPU::S_GETPC_B64).Size == 4);

  // s_getpc_b64 returns the address of next instruction.
  const MCConstantExpr *One = MCConstantExpr::create(4, Ctx);
  SrcBBSym = MCBinaryExpr::createAdd(SrcBBSym, One, Ctx);

  if (MO.getTargetFlags() == OPUInstrInfo::MO_LONG_BRANCH_FORWARD)
    return MCBinaryExpr::createSub(DestBBSym, SrcBBSym, Ctx);

  assert(MO.getTargetFlags() == OPUInstrInfo::MO_LONG_BRANCH_BACKWARD);
  return MCBinaryExpr::createSub(SrcBBSym, DestBBSym, Ctx);
}

bool OPUMCInstLower::lowerOperand(const MachineOperand &MO,
                                     MCOperand &MCOp)  const {
  switch (MO.getType()) {
  default:
    llvm_unreachable("OPUMCInstLower: unknown operand type");
  case MachineOperand::MO_FPImmediate:
    const ConstantFP *Imm = MO.getFPImm();
    APInt Val = Imm->getValueAPF().bitcastToAPInt();
    MCOp = MCOperand::createImm(Val.getZExtValue());
    return true;
  case MachineOperand::MO_Immediate:
    MCOp = MCOperand::createImm(MO.getImm());
    return true;
  case MachineOperand::MO_Register:
    MCOp = MCOperand::createReg(MO.getReg());
    return true;
  case MachineOperand::MO_MachineBasicBlock: {
    MCOp = lowerSymbolOperand(MO, MO.getMBB()->getSymbol());
    /* TODO merged with RISCV
    if (MO.getTargetFlags() != 0) {
      MCOp = MCOperand::createExpr(
        getLongBranchBlockExpr(*MO.getParent()->getParent(), MO));
    } else {
      MCOp = MCOperand::createExpr(
        MCSymbolRefExpr::create(MO.getMBB()->getSymbol(), Ctx));
    }
    */

    return true;
  }
  case MachineOperand::MO_GlobalAddress: {
/* TODO OPU is below AMD is equal to RISCV
    const GlobalValue *GV = MO.getGlobal();
    SmallString<128> SymbolName;
    AP.getNameWithPrefix(SymbolName, GV);
    MCSymbol *Sym = Ctx.getOrCreateSymbol(SymbolName);
    const MCExpr *Expr =
      MCSymbolRefExpr::create(Sym, getVariantKind(MO.getTargetFlags()),Ctx);
    int64_t Offset = MO.getOffset();
    if (Offset != 0) {
      Expr = MCBinaryExpr::createAdd(Expr,
                                     MCConstantExpr::create(Offset, Ctx), Ctx);
    }
    MCOp = MCOperand::createExpr(Expr);
    */
    MCOp = lowerSymbolOperand(MO, AP.getSymbol(MO.getGlobal()));
    return true;
  }
  case MachineOperand::MO_ExternalSymbol: {
    MCSymbol *Sym = Ctx.getOrCreateSymbol(StringRef(MO.getSymbolName()));
    Sym->setExternal(true);
    const MCSymbolRefExpr *Expr = MCSymbolRefExpr::create(Sym,
                                            getVariantKind(MO.getTargetFlags()),
                                            Ctx);
    MCOp = MCOperand::createExpr(Expr);
/* this is RISCV
    MCOp = lowerSymbolOperand(
        MO, AP.GetExternalSymbolSymbol(MO.getSymbolName()), AP);
        */
    return true;
  }
  case MachineOperand::MO_RegisterMask:
    // Regmasks are like implicit defs.
    return false;
  case MachineOperand::MO_ConstantPoolIndex:
    MCOp = lowerSymbolOperand(MO, AP.GetCPISymbol(MO.getIndex()));
    break;
  }
}


/* TODO Use AMD lowerOperand instead
bool llvm::LowerOPUMachineOperandToMCOperand(const MachineOperand &MO,
                                               MCOperand &MCOp,
                                               const AsmPrinter &AP) {
  switch (MO.getType()) {
  default:
    report_fatal_error("LowerOPUMachineInstrToMCInst: unknown operand type");
  case MachineOperand::MO_Register:
    // Ignore all implicit register operands.
    if (MO.isImplicit())
      return false;
    MCOp = MCOperand::createReg(MO.getReg());
    break;
  case MachineOperand::MO_RegisterMask:
    // Regmasks are like implicit defs.
    return false;
  case MachineOperand::MO_Immediate:
    MCOp = MCOperand::createImm(MO.getImm());
    break;
  case MachineOperand::MO_MachineBasicBlock:
    MCOp = lowerSymbolOperand(MO, MO.getMBB()->getSymbol(), AP);
    break;
  case MachineOperand::MO_GlobalAddress:
    MCOp = lowerSymbolOperand(MO, AP.getSymbol(MO.getGlobal()), AP);
    break;
  case MachineOperand::MO_BlockAddress:
    MCOp = lowerSymbolOperand(
        MO, AP.GetBlockAddressSymbol(MO.getBlockAddress()), AP);
    break;
  case MachineOperand::MO_ExternalSymbol:
    MCOp = lowerSymbolOperand(
        MO, AP.GetExternalSymbolSymbol(MO.getSymbolName()), AP);
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    MCOp = lowerSymbolOperand(MO, AP.GetCPISymbol(MO.getIndex()), AP);
    break;
  }
  return true;
}
*/

void OPUMCInstLower::lower(const MachineInstr *MI, MCInst &OutMI) const {
  unsigned Opcode = MI->getOpcode();
  const auto *TII = static_cast<const OPUInstrInfo*>(ST.getInstrInfo());

  // FIXME: Should be able to handle this with emitPseudoExpansionLowering. We
  // need to select it to the subtarget specific version, and there's no way to
  // do that with a single pseudo source operation.
  if (Opcode == OPU::OPU_CALL) {
    OutMI.setOpcode(OPU::S_LCALL);
    MCOperand Dest, Src;
    lowerOperand(MI->getOperand(0), Dest);
    lowerOperand(MI->getOperand(1), Src);
    OutMI.addOperand(Dest);
    OutMI.addOperand(Src);
    return;
  } else if (Opcode == OPU::OPU_TCRETURN) {
    Opcode = OPU::S_JUMP;
  } else if (Opcode == OPU::SIMT_JUMP) {
    Opcode = OPU::SIMT_ACBR_T;
  }

  OutMI.setOpcode(MCOpcode);

  for (const MachineOperand &MO : MI->explicit_operands()) {
    MCOperand MCOp;
    lowerOperand(MO, MCOp);
    OutMI.addOperand(MCOp);
  }
}
/*
void llvm::LowerOPUMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                          const AsmPrinter &AP) {
  OutMI.setOpcode(MI->getOpcode());

  for (const MachineOperand &MO : MI->operands()) {
    MCOperand MCOp;
    if (LowerOPUMachineOperandToMCOperand(MO, MCOp, AP))
      OutMI.addOperand(MCOp);
  }
}
*/
