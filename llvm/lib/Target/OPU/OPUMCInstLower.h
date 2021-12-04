#pragma once

#include "OPUAsmPrinter.h"
#include "OPUSubtarget.h"
#include "OPUTargetMachine.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCStreamer.h"


using namespace llvm;

namespace llvm {

class OPUMCInstLower {
  MCContext &Ctx;
  const TargetSubtargetInfo &ST;
  const AsmPrinter &AP;

  const MCExpr *getLongBranchBlockExpr(const MachineBasicBlock &SrcBB,
                                       const MachineOperand &MO) const;

public:
  OPUMCInstLower(MCContext &ctx, const TargetSubtargetInfo &ST,
                    const AsmPrinter &AP);

  bool lowerOperand(const MachineOperand &MO, MCOperand &MCOp) const;

  /// Lower a MachineInstr to an MCInst
  void lower(const MachineInstr *MI, MCInst &OutMI) const;

  MCOperand lowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym) const;

};
}

