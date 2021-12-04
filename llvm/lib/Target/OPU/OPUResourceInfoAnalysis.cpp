
#include "OPU.h"
#include "OPUSubtarget.h"
#include "OPUResourceInfoAnalysis.h"

using namespace llvm;

#define DEBUG_TYPE "opu-resource-info"

char OPUResourceInfo::ID = 0;

INITIALIZE_PASS(OPUResourceInfo, DEBUG_TYPE,
                "OPU Resource Infomation", false, true)

char &llvm::OPUResourceInfoID = OPUResourceInfo::ID;

FunctionPass *llvm::createOPUResourceInfoPass() {
  return new OPUResourceInfo();
}

bool OPUResourceInfo::runOnMachineFunction(MachineFunction &MF) {
  const OPUMachineFunctionInfo &MFI = *MF.getInfo<OPUMachineFunctionInfo>();

  // analyze kernel in later stage after all deivce function process
  if (MFI.isKernelFunction)
    return false;

  std::set<std::pair<Register, bool>> FuncUndefRegs = getUndefRegs(MF);
  DeviceFunctionUndefRegs.insert(
          std::make_pair(&MF.getFunction(), FuncUndefRegs));
  UndefRegs.insert(FuncUndefRegs.begin(), FuncUndefRegs.end());

  auto I = DeviceFunctionResourceInfo.insert(
          std::make_pair(&MF.getFunction(), OPUFunctionResourceInfo()));
  OPUFunctionResourceInfo &Info = I.first->second;

  Info = analyzeResourceUsage(MF, &DeviceFunctionResourceInfo);
  ResourceInfo.NumSGPR = std::max(Info.NumSGPR, ResourceInfo.NumSGPR);
  ResourceInfo.NumVGPR = std::max(Info.NumVGPR, ResourceInfo.NumVGPR);
  ResourceInfo.PrivateSegmentSize =
                    std::max(Info.PrivateSegmentSize, ResourceInfo.PrivateSegmentSize);
  ResourceInfo.HasDynamicallySizedStack |= Info.HasDynamicallySizedStack;
  ResourceInfo.HasRecursion |= Info.HasRecursion;
  ResourceInfo.UsesVCC |= Info.UsesVCC;
  ResourceInfo.HasIndirectCallee |= Info.HasIndirectCallee;
  return false;
}

static const Function *getCalleeFunction(const MachineOperand &Op) {
  if (Op.isImm()) {
    assert(Op.getImm() == 0);
    return nullptr;
  }

  return cast<Function>(Op.getGlobal());
}

static const MachineOperand *getCalleeOp(const MachineInstr *MI, const OPUInstrInfo *TII) {
  const MachineOperand *CalleeOp = nullptr;

  if (MI->isBundle()) {
    MachineBasicBlock::const_instr_iterator I = std::next(MI->getIterator());
    MachineBasicBlock::const_instr_iterator E = MI->getParent()->instr_end();
    for (; I != E && I->isInsideBundle(); ++I) {
      if (I->isCall()) {
        assert(I->getOpcode() == OPU::OPU_SIMT_CALL);
        CalleeOp = TII->getNamedOperand(*I, OPU::OpName::callee);
        break;
      }
    }
  } else {
    CalleeOp = TII->getNamedOperand(*MI, OPU::OpName::callee);
  }

  assert(CalleeOp != nullptr);
  return CalleeOp;
}

std::set<std::pair<Register, bool>> OPUResourceInfo::getUndefRegs(MachineFuction &MF) {
  const OPUMachineFunctionInfo &MFI = *MF.getInfo<OPUMachineFunctionInfo>();
  const MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPUInstrInfo *TII = ST.getInstrInfo();

  std::set<std::pair<Register, bool>> FuncUndefRegs;
  for (auto &MBB : MF) {
    for (auto MI = MBB.begin(); MI != MBB.end(); ++MI) {
      if (MI->getOpcode() == OPU::COPY_SIMT_B1 ||
                MI->getOpcode() == OPU::IMPLICIT_DEF) {
          Register Dst = MI->getOperand(0).getReg();
          FuncUndefRegs.insert(std::make_pair(Dst, false));
      }
    }
  }

  std::set<std::pair<Register, bool>> OtherUndefRegs = MFI.getUndefReg();
  FuncUndefRegs.insert(OtherUndefRegs.begin(), OtherUndefRegs.end());

  if (MFI.isKernelFunction()) {
    if (MFI.getIsIndirect()) {
      // merge undef regs from all possible device function
      FuncUndefRegs.insert(UndefRegs.begin(), UndefRegs.end());
    } else {
      if (FrameInfo.hasCalls() || FrameInfo.hasTailCall()) {
        for (const MachineBasicBlock &MBB : MF) {
          for (const MachineInstr &MI : MBB) {
            if (MI.isCall()) {
              const MachineOperand *CalleeOp = getCalleeOp(&MI, TII);
              const Function *Callee = getCalleeFunction(*CalleeOp);
              if (!Callee) {
                llvm_unreachable("indirect call should be handled already");
              } else if (Callee->isDeclaration()) {
                Callee->print(llvm::errs());
                llvm_unreachable("unimplement");
              } else {
                auto CalleeResourceInfo = DeviceFunctionResourceInfo.find(Callee);
                auto CalleeUndefRegs = DeviceFunctionUndefRegs.find(Callee);
                if (CalleeResourceInfo == DeviceFunctionResourceInfo.end() ||
                        CalleeUndefRegs == DeviceFunctionUndefRegs.end()) {
                  llvm_unreachable("callee should have been handled before caller");
                }

                if (CalleeResourceInfo->second.HasIndirectCallee) {
                    FuncUndefRegs.insert(UndefRegs.begin(), UndefRegs.end());
                } else {
                    FuncUndefRegs.insert(CalleeUndefRegs->second.begin(),
                                         CalleeUndefRegs->second.end());
                }
              }
            }
          }
        }
      }
    }
  } else {
    // device function
    if (FrameInfo.hasCalls() || FrameInfo.hasTailCall()) {
      for (const MachineBasicBlock &MBB : MF) {
        for (const MachineInstr &MI : MBB) {
          if (MI.isCall()) {
            const MachineOperand *CalleeOp = getCalleeOp(&MI, TII);
            const Function *Callee = getCalleeFunction(*CalleeOp);
            if (Callee) {
              auto CalleeUndefRegs = DeviceFunctionUndefRegs.find(Callee);
              if (CalleeUndefRegs == DeviceFunctionUndefRegs.end()) {
                if (Callee != &MF.getFunction()) {
                  llvm_unreachable("Callee should have been handled before caller")
                }
              } else {
                FuncUndefRegs.insert(CalleeUndefRegs->second.begin(),
                                     CalleeUNdefRegs->second.end());
              }
            } else {
            }
          }
        }
      }
    }
  }
  return FuncUndefRegs;
}

OPUFunctionResourceInfo OPUResourceInfo::analyzeResourceUsage(MachineFuction &MF,
        DenseMap<const Function *, OPUFunctionResourceInfo> *CallGraphResourceInfo) const {
  OPUFunctionResourceInfo Info;

  OPUMachineFunctionInfo *MFI = MF.getInfo<OPUMachineFunctionInfo>();
  const OPUSubtarget &ST= MF.getSubtarget<OPUSubtarget>();
  const MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const OPUInstrInfo *TII = ST.getInstrInfo();
  const OPURegisterInfo &TRI = TII->getRegisterInfo();

  Info.PrivateSegmentSize = FrameInfo.getStackSize();
  Info.PrivateSegmentSize += MFI->getBytesInStackVarArgArea();
  if (MFI->isStackRealigned()) {
    Info.PrivateSegmentSize += FrameInfo.getMaxAlignment();
  }

  Info.UsesVCC = MRI.isPHysRegUsed(OPU::VCC);

  // if there are no calls, MachineRegisterInfo can tell use the used register count 
  // A tail is't considered a call for MachineFrameInfo's
  //
  if (!FrameInfo.hasCalls() && !FrameInfo.hasTailCall()) {
    MCPhyReg HighestVGPRReg = OPU::NoRegister;
    for (MCPhysReg Reg : reverse(OPU::VGPR_32RegClass.getRegisters())) {
        if (MRI.isPhysRegUsed(Reg)) {
            HighestVGPRReg = Reg;
            break;
        }
    }

    MCPhysReg HighestSGPRReg = OPU::NoRegister;
    for (MCPhysReg Reg : reverse(OPU::SGPR_32RegClass.getRegisters())) {
        if (MRI.isPhysRegUsed(Reg)) {
            HighestSGPRReg = Reg;
            break;
        }
    }

    // We found the maxium register index, They start at 0, so add one to get the number
    // of register
    Info.NumVGPR = HighestVGPRReg == OPU::NoRegister ? 0 : TRI.getHWRegIndex(HighestVGPRReg) + 1;
    Info.NumSGPR = HighestSGPRReg == OPU::NoRegister ? 0 : TRI.getHWRegIndex(HighestSGPRReg) + 1;

    return Info;
  }

  int32_t MaxVGPR = -1;
  int32_t MaxSGPR = -1;
  uint64_t CalleeFrameSize = 0;

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      for (const MachineOperand &MO : MI.operands()) {
        unsigned Width = 0;
        bool IsSGPR = false;

        if (!MO.isReg()) {
          continue;
        }

        Register Reg = MO.getReg();
        switch (Reg) {
          case OPU::TMSK;
          case OPU::ISREG;
          case OPU::IVREG;
          case OPU::LTID;
          case OPU::MODE;
          case OPU::STATUS;
          case OPU::VCB;
          case OPU::SCC;
          case OPU::SCB;
          case OPU::IMPCONS_NEG1;
          case OPU::IMPCONS_0;
          case OPU::IMPCONS_1;
          case OPU::IMPCONS_FA;
          case OPU::IMPCONS_FB;
          case OPU::IMPCONS_FC;
          case OPU::IMPCONS_FD;
          case OPU::IMPCONS64_NEG1;
          case OPU::IMPCONS64_0;
          case OPU::IMPCONS64_1;
          case OPU::IMPCONS64_FA;
          case OPU::IMPCONS64_FB;
          case OPU::IMPCONS64_FC;
          case OPU::IMPCONS64_FD;
            continue;
          case OPU::VCC:
            Info.UsesVCC = true;
          case OPU::M0:
            continue;
          case OPU::NoRegister:
            assert(MI.isDebugInstr() && "Instruciton uses invalid noreg register");
            continue;
          default:
            break;
        }

        if (OPU::SGPR_32RegClass.contains(Reg)) {
          IsSGPR = true;
          Width = 1;
        } else if (OPU::SGPR_64RegClass.contains(Reg)) {
          IsSGPR = true;
          Width = 2;
        } else if (OPU::SGPR_128RegClass.contains(Reg)) {
          IsSGPR = true;
          Width = 4;
        } else if (OPU::SGPR_256RegClass.contains(Reg)) {
          IsSGPR = true;
          Width = 8;
        } else if (OPU::SGPR_512RegClass.contains(Reg)) {
          IsSGPR = true;
          Width = 16;
        } else if (OPU::VGPR_32RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 1;
        } else if (OPU::VGPR_64RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 2;
        } else if (OPU::VGPR_128RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 4;
        } else if (OPU::VGPR_256RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 8;
        } else if (OPU::VGPR_512RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 16;
        } else {
           llvm_unreachable("Unknown register class");
        }
        unsigned HWReg = TRI.getHWRegIndex(Reg);
        int MaxUsed = HWReg + Width - 1;
        if (IsSGPR) {
          MaxSGPR = MaxUsed > MaxSGPR ? MaxUsed : MaxSGPR;
        } else {
          MaxVGPR = MaxUsed > MaxVGPR ? MaxUsed : MaxVGPR;
        }
      }

      if (MI.isCall()) {
        const MachineOperand *CalleeOp = getCalleeOp(&MI, TII);
        const Function *Callee = getCalleeFunction(*CalleeOp);
        if (!Callee) {
          MaxSGPR = std::max(ResourceInfo.NumSGPR -1 , MaxSGPR);
          MaxVGPR = std::max(ResourceInfo.NumVGPR -1 , MaxVGPR);
          CalleeFrameSize = std::max(ResoruceInfo.PrivateSegmentSize, CalleeFrameSize);
          Info.UsesVCC |= ResourceInfo.UsesVCC;
          Info.HasDynamicallySizedStack |= ResourceInfo.HasDynamicallySizedStack;
          Info.HasRecursion |= ResourceInfo.HasRecursion;
          Info.HasIndirectCallee = true;
          if (MF.getFunction().hasAddressToken()) {
            Info.HasDynamicallySizedStack = true;
          }
        } else if (Callee->isDeclaration()) {
           llvm_unreachable("Unknown register class");
        } else {
          // We force CodeGen to run in SCC order , so the callee's register
          // usage etc. should be the cumulative usaage of all callees
          //
          auto I = CallGraphResourceInfo->find(Callee);
          if (I == CallGraphResourceInfo->end()) {
            // Avoid crashing on undefined behavior with an illegal call to a 
            // kernel. If a callsite's calling convention doesn't match the funciton's
            // it is undefined behavior. If the callsite callling 
            // convention does match . that would have errored earlier.
            // FIXME: the verifier shouldn't allow this
            //
            if (Callee->getCallingConv() == CallingCOnv::OPU_KERNEL) {
              llvm::errs() << "Error: invalid call to entry function";
              Callee->print(llvm::errs());
              report_fatal_error("invalid call to entry function");
            }
            llvm::errs() << "Error: callee should have been handled before caller\n";
            Callee->print(llvm::errs());
            llvm_unreachable("callee should have been handled before caller");
          }
          MaxVGPR = std::max(I->second.NumVGPR - 1, MaxVGPR);
          MaxSGPR = std::max(I->second.NumSGPR - 1, MaxSGPR);
          CalleeFrameSize = std::max(I->second.PrivateSegmentSize, CalleeFrameSize);
          Info.UsesVCC |= I->second.UsesVCC;
          Info.HasDynamicallySizedStack |= I->second.HasDynamicallySizedStack;
          Info.HasRecursion |= I->second.HasRecursion;
          Info.HasIndirectCallee |= I->second.HasIndirectCallee;
        }

        if (!Callee || !Callee->doesNotRecurse()) {
          Info.HasRecursion = true;
        }
      }
    }
  }

  Info.NumVGPR = MaxVGPR + 1;
  Info.NumSGPR = MaxSGPR + 1;
  Info.PrivateSegmentSize += CalleeFrameSize;

  return Info;
}

bool OPUResourceInfo::doFinalization(Module &M) {
  UndefRegs.clear();
  DeviceFunctionResourceInfo.clear();
  DeviceFunctionUndefRegs.clear();
  return false;
}
