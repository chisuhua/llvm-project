//===- OPUMemoryLegalizer.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Memory legalizer - implements memory model. More information can be
/// found here:
///   http://llvm.org/docs/OPUUsage.html#memory-model
//
//===----------------------------------------------------------------------===//

#include "OPU.h"
#include "OPUMachineModuleInfo.h"
#include "OPUSubtarget.h"
#include "OPUDefines.h"
#include "OPUInstrInfo.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "Utils/OPUBaseInfo.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Pass.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <list>

using namespace llvm;
using namespace llvm::OPU;

#define DEBUG_TYPE "ppu-memory-legalizer"
#define PASS_NAME "OPU Memory Legalizer"

namespace {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

/// Memory operation flags. Can be ORed together.
enum class OPUMemOp {
  NONE = 0u,
  LOAD = 1u << 0,
  STORE = 1u << 1,
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestFlag = */ STORE)
};

/// Position to insert a new instruction relative to an existing
/// instruction.
enum class Position {
  BEFORE,
  AFTER
};

/// The atomic synchronization scopes supported by the OPU target.
enum class OPUAtomicScope {
  NONE,
  SINGLETHREAD,
  BLOCK,
  DEVICE,
  SYSTEM
};

enum class OPUCacheBit {
  NONE,
  GLC,
}

/// The distinct address spaces supported by the OPU target for
/// atomic memory operation. Can be ORed toether.
enum class OPUAtomicAddrSpace {
  NONE = 0u,
  FLAT = 1u << 0,
  GLOBAL = 1u << 1,
  SHARED = 1u << 2,
  PRIVATE = 1u << 4,
  OTHER = 1u << 8,

  /// The address spaces that can be accessed by a FLAT instruction.
  FLAT = GLOBAL | SHARED | SCRATCH,

  /// The address spaces that support atomic instructions.
  ATOMIC = GLOBAL | SHARED | SCRATCH,

  /// All address spaces.
  ALL = GLOBAL | SHARED | SCRATCH | OTHER,

  LLVM_MARK_AS_BITMASK_ENUM(/* LargestFlag = */ ALL)
};

/// Sets named bit \p BitName to "true" if present in instruction \p MI.
/// \returns Returns true if \p MI is modified, false otherwise.
template <uint16_t BitName>
bool enableNamedBit(const MachineBasicBlock::iterator &MI) {
  int BitIdx = OPU::getNamedOperandIdx(MI->getOpcode(), BitName);
  if (BitIdx == -1)
    return false;

  MachineOperand &Bit = MI->getOperand(BitIdx);
  if (Bit.getImm() != 0)
    return false;

  Bit.setImm(1);
  return true;
}

class OPUMemOpInfo final {
private:

  friend class OPUMemOpAccess;

  AtomicOrdering Ordering = AtomicOrdering::NotAtomic;
  AtomicOrdering FailureOrdering = AtomicOrdering::NotAtomic;
  OPUAtomicScope Scope = OPUAtomicScope::SYSTEM;
  OPUAtomicAddrSpace OrderingAddrSpace = OPUAtomicAddrSpace::NONE;
  OPUAtomicAddrSpace InstrAddrSpace = OPUAtomicAddrSpace::NONE;
  bool IsCrossAddressSpaceOrdering = false;
  bool IsVolatile = false;

  OPUMemOpInfo(AtomicOrdering Ordering = AtomicOrdering::SequentiallyConsistent,
              OPUAtomicScope Scope = OPUAtomicScope::SYSTEM,
              OPUAtomicAddrSpace OrderingAddrSpace = OPUAtomicAddrSpace::ATOMIC,
              OPUAtomicAddrSpace InstrAddrSpace = OPUAtomicAddrSpace::ALL,
              AtomicOrdering FailureOrdering = AtomicOrdering::SequentiallyConsistent,
              bool IsCrossAddressSpaceOrdering = true,
              bool IsVolatile = false)
    : Ordering(Ordering), FailureOrdering(FailureOrdering),
      Scope(Scope), OrderingAddrSpace(OrderingAddrSpace),
      InstrAddrSpace(InstrAddrSpace),
      IsCrossAddressSpaceOrdering(IsCrossAddressSpaceOrdering),
      IsVolatile(IsVolatile) {
    // There is also no cross address space ordering if the ordering
    // address space is the same as the instruction address space and
    // only contains a single address space.
    if ((OrderingAddrSpace == InstrAddrSpace) &&
        isPowerOf2_32(uint32_t(InstrAddrSpace)))
      IsCrossAddressSpaceOrdering = false;
  }

public:
  /// \returns Atomic synchronization scope of the machine instruction used to
  /// create this OPUMemOpInfo.
  OPUAtomicScope getScope() const {
    return Scope;
  }

  /// \returns Ordering constraint of the machine instruction used to
  /// create this OPUMemOpInfo.
  AtomicOrdering getOrdering() const {
    return Ordering;
  }

  /// \returns Failure ordering constraint of the machine instruction used to
  /// create this OPUMemOpInfo.
  AtomicOrdering getFailureOrdering() const {
    return FailureOrdering;
  }

  /// \returns The address spaces be accessed by the machine
  /// instruction used to create this SiMemOpInfo.
  OPUAtomicAddrSpace getInstrAddrSpace() const {
    return InstrAddrSpace;
  }

  /// \returns The address spaces that must be ordered by the machine
  /// instruction used to create this SiMemOpInfo.
  OPUAtomicAddrSpace getOrderingAddrSpace() const {
    return OrderingAddrSpace;
  }

  /// \returns Return true iff memory ordering of operations on
  /// different address spaces is required.
  bool getIsCrossAddressSpaceOrdering() const {
    return IsCrossAddressSpaceOrdering;
  }

  /// \returns True if memory access of the machine instruction used to
  /// create this OPUMemOpInfo is non-temporal, false otherwise.
  bool isVolatile() const {
    return IsVolatile;
  }


  /// \returns True if ordering constraint of the machine instruction used to
  /// create this OPUMemOpInfo is unordered or higher, false otherwise.
  bool isAtomic() const {
    return Ordering != AtomicOrdering::NotAtomic;
  }

};

class OPUMemOpAccess final {
private:
  OPUMachineModuleInfo *MMI = nullptr;

  /// Reports unsupported message \p Msg for \p MI to LLVM context.
  void reportUnsupported(const MachineBasicBlock::iterator &MI,
                         const char *Msg) const;

  /// Inspects the target synchonization scope \p SSID and determines
  /// the OPU atomic scope it corresponds to, the address spaces it
  /// covers, and whether the memory ordering applies between address
  /// spaces.
  Optional<std::tuple<OPUAtomicScope, OPUAtomicAddrSpace, bool>>
  toOPUAtomicScope(SyncScope::ID SSID, OPUAtomicAddrSpace InstrScope) const;

  /// \return Return a bit set of the address spaces accessed by \p AS.
  OPUAtomicAddrSpace toOPUAtomicAddrSpace(unsigned AS) const;

  /// \returns Info constructed from \p MI, which has at least machine memory
  /// operand.
  Optional<OPUMemOpInfo> constructFromMIWithMMO(
      const MachineBasicBlock::iterator &MI) const;

public:
  /// Construct class to support accessing the machine memory operands
  /// of instructions in the machine function \p MF.
  OPUMemOpAccess(MachineFunction &MF);

  /// \returns Load info if \p MI is a load operation, "None" otherwise.
  Optional<OPUMemOpInfo> getLoadInfo(
      const MachineBasicBlock::iterator &MI) const;

  /// \returns Store info if \p MI is a store operation, "None" otherwise.
  Optional<OPUMemOpInfo> getStoreInfo(
      const MachineBasicBlock::iterator &MI) const;

  /// \returns Atomic fence info if \p MI is an atomic fence operation,
  /// "None" otherwise.
  Optional<OPUMemOpInfo> getAtomicFenceInfo(
      const MachineBasicBlock::iterator &MI) const;

  /// \returns Atomic cmpxchg/rmw info if \p MI is an atomic cmpxchg or
  /// rmw operation, "None" otherwise.
  Optional<OPUMemOpInfo> getAtomicCmpxchgOrRmwInfo(
      const MachineBasicBlock::iterator &MI) const;
};

class OPUCacheControl {
protected:

  /// Instruction info.
  const OPUInstrInfo *TII = nullptr;

  IsaInfo::IsaVersion IV;
  // IsaVersion IV;

  OPUCacheControl(const OPUSubtarget &ST);

  bool setCachePolicy(const MachineBasicBlock::iterator &MI, unsigned CachePolicy) const;

public:

  /// Create a cache control for the subtarget \p ST.
  static std::unique_ptr<OPUCacheControl> create(const OPUSubtarget &ST) {
    return std::make_unique<OPUCacheControl>(ST);
  }

  /// Update \p MI memory load instruction to bypass any caches up to
  /// the \p Scope memory scope for address spaces \p
  /// AddrSpace. Return true iff the instruction was modified.
  bool enableLoadCacheBypass(const MachineBasicBlock::iterator &MI,
                                     OPUAtomicScope Scope,
                                     OPUAtomicAddrSpace AddrSpace) const = 0;

  /// Update \p MI memory instruction to indicate it is
  /// nontemporal. Return true iff the instruction was modified.
  bool enableVolatile(const MachineBasicBlock::iterator &MI)

  bool insertCacheEvict(MachineBasicBlock::iterator &MI,
                                     OPUAtomicScope Scope,
                                     OPUAtomicAddrSpace AddrSpace,
                                     Position Pos) const = 0;

  /// Inserts any necessary instructions at position \p Pos relative
  /// to instruction \p MI to ensure any caches associated with
  /// address spaces \p AddrSpace for memory scopes up to memory scope
  /// \p Scope are invalidated. Returns true iff any instructions
  /// inserted.
  bool insertCacheInvalidate(MachineBasicBlock::iterator &MI,
                                     OPUAtomicScope Scope,
                                     OPUAtomicAddrSpace AddrSpace,
                                     Position Pos) const = 0;

  /// Inserts any necessary instructions at position \p Pos relative
  /// to instruction \p MI to ensure memory instructions of kind \p Op
  /// associated with address spaces \p AddrSpace have completed as
  /// observed by other memory instructions executing in memory scope
  /// \p Scope. \p IsCrossAddrSpaceOrdering indicates if the memory
  /// ordering is between address spaces. Returns true iff any
  /// instructions inserted.
  bool insertWait(MachineBasicBlock::iterator &MI,
                          OPUAtomicScope Scope,
                          OPUAtomicAddrSpace AddrSpace,
                          OPUMemOp Op,
                          bool IsCrossAddrSpaceOrdering,
                          Position Pos) const = 0;

  bool insertFence(MachineBasicBlock::iterator &MI,
                          OPUAtomicScope Scope,
                          OPUAtomicAddrSpace AddrSpace,
                          OPUMemOp Op,
                          bool IsCrossAddrSpaceOrdering,
                          Position Pos) const = 0;

  bool insertWaitFence(MachineBasicBlock::iterator &MI,
                          OPUAtomicScope Scope,
                          OPUAtomicAddrSpace AddrSpace,
                          OPUMemOp Op,
                          bool IsCrossAddrSpaceOrdering,
                          Position Pos) const = 0;
};

class SIGfx6CacheControl : public OPUCacheControl {
protected:

  /// Sets GLC bit to "true" if present in \p MI. Returns true if \p MI
  /// is modified, false otherwise.
  bool enableGLCBit(const MachineBasicBlock::iterator &MI) const {
    return true;
    //return enableNamedBit<OPU::OpName::glc>(MI);
  }

  /// Sets SLC bit to "true" if present in \p MI. Returns true if \p MI
  /// is modified, false otherwise.
  bool enableSLCBit(const MachineBasicBlock::iterator &MI) const {
    // FIXME return enableNamedBit<OPU::OpName::slc>(MI);
    return false;
  }

  /*
  bool enableKOPBit(const MachineBasicBlock::iterator &MI, unsigned imm) const {
    return enableNamedBit<OPU::OpName::kop>(MI);
  }
  */

public:

  SIGfx6CacheControl(const OPUSubtarget &ST) : OPUCacheControl(ST) {};

  bool enableLoadCacheBypass(const MachineBasicBlock::iterator &MI,
                             OPUAtomicScope Scope,
                             OPUAtomicAddrSpace AddrSpace) const override;

  bool enableVolatile(const MachineBasicBlock::iterator &MI) const override;

  bool insertCacheInvalidate(MachineBasicBlock::iterator &MI,
                             OPUAtomicScope Scope,
                             OPUAtomicAddrSpace AddrSpace,
                             Position Pos) const override;

  bool insertWait(MachineBasicBlock::iterator &MI,
                  OPUAtomicScope Scope,
                  OPUAtomicAddrSpace AddrSpace,
                  OPUMemOp Op,
                  bool IsCrossAddrSpaceOrdering,
                  Position Pos) const override;
};

class SIGfx7CacheControl : public SIGfx6CacheControl {
public:

  SIGfx7CacheControl(const OPUSubtarget &ST) : SIGfx6CacheControl(ST) {};

  bool insertCacheInvalidate(MachineBasicBlock::iterator &MI,
                             OPUAtomicScope Scope,
                             OPUAtomicAddrSpace AddrSpace,
                             Position Pos) const override;

};

class OPUSyncControl final {
protected:
  // <MI, Flag> Flag is true when MI is a strong BlkSyn
  DenseMap<MachineInstr *, bool> BlkSynMIs;
  // <MI, Flag> Flag is uselss , always is true
  DenseMap<MachineInstr *, bool> CrossScopeMIs;
  // <MI, Flag> Flag is uselss , always is true
  SmallDenseMap<MachineBasicBlock *, bool> BlkSynMBBs;
  // <MI, Flag> Flag is true when where is no blysync before first CrossScopeMI
  SmallDenseMap<MachineBasicBlock *, bool> CrossScopeMBBs;

  const OPUInstrInfo *TII = nullptr;
public:
  OPUSyncControl(const OPUSubtarget &ST);

  static std::unique_ptr<OPUSyncControl> create(const OPUSubtarget &ST) {
    return std::make_unique<OPUSyncControl>(ST);
  }

  void insertBlkSynMIs(MachineInstr *MI) {
    BlkSynMIs.insert(std::make_pair(MI, false));
    BlkSynMBBs.insert(std::make_pair(MI->getParent(), true));
  }

  void insertCrossScopeMIs(MachineInstr *MI) {
    CrossScopeMIs.insert(std::make_pair(MI, false));
    CrossScopeMBBs.insert(std::make_pair(MI->getParent(), true));
  }

  void analysisBlkSyn();
  bool reachSuccCrossScopeMBB(MachineInstr *MI);
  bool updateBlkSyn();
}

class OPUMemoryLegalizer final : public MachineFunctionPass {
private:
  const OPUInstrInfo *TII = nullptr;

  /// Cache Control.
  std::unique_ptr<OPUCacheControl> CC = nullptr;

  // BlkSync Control
  std::unique_ptr<OPUSyncControl> SC = nullptr;

  /// List of atomic pseudo instructions.
  std::list<MachineBasicBlock::iterator> AtomicPseudoMIs;

  /// Return true iff instruction \p MI is a atomic instruction that
  /// returns a result.
  bool isAtomicRet(const MachineInstr &MI) const {
    // FIXME return OPU::getAtomicNoRetOp(MI.getOpcode()) != -1;
    return OPU::getAtomicNoRetOp(MI.getOpcode()) != -1;
  }

  /// Removes all processed atomic pseudo instructions from the current
  /// function. Returns true if current function is modified, false otherwise.
  bool removeAtomicPseudoMIs();

  /// Expands load operation \p MI. Returns true if instructions are
  /// added/deleted or \p MI is modified, false otherwise.
  bool expandLoad(const OPUMemOpInfo &MOI,
                  MachineBasicBlock::iterator &MI);
  /// Expands store operation \p MI. Returns true if instructions are
  /// added/deleted or \p MI is modified, false otherwise.
  bool expandStore(const OPUMemOpInfo &MOI,
                   MachineBasicBlock::iterator &MI);
  /// Expands atomic fence operation \p MI. Returns true if
  /// instructions are added/deleted or \p MI is modified, false otherwise.
  bool expandAtomicFence(const OPUMemOpInfo &MOI,
                         MachineBasicBlock::iterator &MI);
  /// Expands atomic cmpxchg or rmw operation \p MI. Returns true if
  /// instructions are added/deleted or \p MI is modified, false otherwise.
  bool expandAtomicCmpxchgOrRmw(const OPUMemOpInfo &MOI,
                                MachineBasicBlock::iterator &MI);

public:
  static char ID;

  OPUMemoryLegalizer() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return PASS_NAME;
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end namespace anonymous

void OPUMemOpAccess::reportUnsupported(const MachineBasicBlock::iterator &MI,
                                      const char *Msg) const {
  const Function &Func = MI->getParent()->getParent()->getFunction();
  DiagnosticInfoUnsupported Diag(Func, Msg, MI->getDebugLoc());
  Func.getContext().diagnose(Diag);
}

Optional<std::tuple<OPUAtomicScope, OPUAtomicAddrSpace, bool>>
OPUMemOpAccess::toOPUAtomicScope(SyncScope::ID SSID,
                               OPUAtomicAddrSpace InstrScope) const {
  if (SSID == SyncScope::System || SSID == MMI->getSystemSSID())
    return std::make_tuple(OPUAtomicScope::SYSTEM,
                           OPUAtomicAddrSpace::ATOMIC,
                           true);
  if (SSID == MMI->getDeviceSSID())
    return std::make_tuple(OPUAtomicScope::DEVICE,
                           OPUAtomicAddrSpace::ATOMIC,
                           true);
  if (SSID == MMI->getBlockSSID())
    return std::make_tuple(OPUAtomicScope::BLOCK,
                           OPUAtomicAddrSpace::ATOMIC,
                           true);
  if (SSID == SyncScope::SingleThread)
    return std::make_tuple(OPUAtomicScope::SINGLETHREAD,
                           OPUAtomicAddrSpace::ATOMIC,
                           true);
  return None;
}

OPUAtomicAddrSpace OPUMemOpAccess::toOPUAtomicAddrSpace(unsigned AS) const {
  if (AS == OPUAS::FLAT_ADDRESS)
    return OPUAtomicAddrSpace::FLAT;
  if (AS == OPUAS::GLOBAL_ADDRESS)
    return OPUAtomicAddrSpace::GLOBAL;
  if (AS == OPUAS::SHARED_ADDRESS)
    return OPUAtomicAddrSpace::SHARED;
  if (AS == OPUAS::PRIVATE_ADDRESS)
    return OPUAtomicAddrSpace::PRIVATE;
  //if (AS == OPUAS::REGION_ADDRESS)
  //  return OPUAtomicAddrSpace::GDS;

  return OPUAtomicAddrSpace::OTHER;
}

OPUMemOpAccess::OPUMemOpAccess(MachineFunction &MF) {
  MMI = &MF.getMMI().getObjFileInfo<OPUMachineModuleInfo>();
}

Optional<OPUMemOpInfo> OPUMemOpAccess::constructFromMIWithMMO(
    const MachineBasicBlock::iterator &MI) const {
  assert(MI->getNumMemOperands() > 0);

  SyncScope::ID SSID = SyncScope::SingleThread;
  AtomicOrdering Ordering = AtomicOrdering::NotAtomic;
  AtomicOrdering FailureOrdering = AtomicOrdering::NotAtomic;
  OPUAtomicAddrSpace InstrAddrSpace = OPUAtomicAddrSpace::NONE;
  bool IsVolatile = true;

  // Validator should check whether or not MMOs cover the entire set of
  // locations accessed by the memory instruction.
  for (const auto &MMO : MI->memoperands()) {
    IsVolatile &= MMO->isVolatile();
    InstrAddrSpace |=
      toOPUAtomicAddrSpace(MMO->getPointerInfo().getAddrSpace());
    AtomicOrdering OpOrdering = MMO->getOrdering();
    if (OpOrdering != AtomicOrdering::NotAtomic) {
      const auto &IsSyncScopeInclusion =
          MMI->isSyncScopeInclusion(SSID, MMO->getSyncScopeID());
      if (!IsSyncScopeInclusion) {
        reportUnsupported(MI,
          "Unsupported non-inclusive atomic synchronization scope");
        return None;
      }

      SSID = IsSyncScopeInclusion.getValue() ? SSID : MMO->getSyncScopeID();
      Ordering =
          isStrongerThan(Ordering, OpOrdering) ?
              Ordering : MMO->getOrdering();
      assert(MMO->getFailureOrdering() != AtomicOrdering::Release &&
             MMO->getFailureOrdering() != AtomicOrdering::AcquireRelease);
      FailureOrdering =
          isStrongerThan(FailureOrdering, MMO->getFailureOrdering()) ?
              FailureOrdering : MMO->getFailureOrdering();
    }
  }

  OPUAtomicScope Scope = OPUAtomicScope::NONE;
  OPUAtomicAddrSpace OrderingAddrSpace = OPUAtomicAddrSpace::NONE;
  bool IsCrossAddressSpaceOrdering = false;
  if (Ordering != AtomicOrdering::NotAtomic) {
    auto ScopeOrNone = toOPUAtomicScope(SSID, InstrAddrSpace);
    if (!ScopeOrNone) {
      reportUnsupported(MI, "Unsupported atomic synchronization scope");
      return None;
    }
    std::tie(Scope, OrderingAddrSpace, IsCrossAddressSpaceOrdering) =
      ScopeOrNone.getValue();
    if ((OrderingAddrSpace == OPUAtomicAddrSpace::NONE) ||
        ((OrderingAddrSpace & OPUAtomicAddrSpace::ATOMIC) != OrderingAddrSpace)) {
      reportUnsupported(MI, "Unsupported atomic address space");
      return None;
    }
  }
  return OPUMemOpInfo(Ordering, Scope, OrderingAddrSpace, InstrAddrSpace,
                     IsCrossAddressSpaceOrdering, FailureOrdering, IsVolatile);
}

Optional<OPUMemOpInfo> OPUMemOpAccess::getLoadInfo(
    const MachineBasicBlock::iterator &MI) const {
  assert(MI->getDesc().TSFlags & OPUInstrFlags::maybeAtomic);

  if (!(MI->mayLoad() && !MI->mayStore()))
    return None;

  // Be conservative if there are no memory operands.
  if (MI->getNumMemOperands() == 0)
    return OPUMemOpInfo(AtomicOrdering::NotAtomic,
                        OPUAtomicScope::SYSTEM,
                        OPUAtomicAddrSpace::ATOMIC,
                        OPUAtomicAddrSpace::ALL,
                        AtomicOrdering::NotAtomic, false, false);

  return constructFromMIWithMMO(MI);
}

Optional<OPUMemOpInfo> OPUMemOpAccess::getStoreInfo(
    const MachineBasicBlock::iterator &MI) const {
  assert(MI->getDesc().TSFlags & OPUInstrFlags::maybeAtomic);

  if (!(!MI->mayLoad() && MI->mayStore()))
    return None;

  // Be conservative if there are no memory operands.
  if (MI->getNumMemOperands() == 0)
    return OPUMemOpInfo(AtomicOrdering::NotAtomic,
                        OPUAtomicScope::SYSTEM,
                        OPUAtomicAddrSpace::ATOMIC,
                        OPUAtomicAddrSpace::ALL,
                        AtomicOrdering::NotAtomic, false);

  return constructFromMIWithMMO(MI);
}

Optional<OPUMemOpInfo> OPUMemOpAccess::getAtomicFenceInfo(
    const MachineBasicBlock::iterator &MI) const {
  assert(MI->getDesc().TSFlags & OPUInstrFlags::maybeAtomic);

  if (MI->getOpcode() != OPU::ATOMIC_FENCE)
    return None;

  AtomicOrdering Ordering =
    static_cast<AtomicOrdering>(MI->getOperand(0).getImm());

  SyncScope::ID SSID = static_cast<SyncScope::ID>(MI->getOperand(1).getImm());
  auto ScopeOrNone = toOPUAtomicScope(SSID, OPUAtomicAddrSpace::ATOMIC);
  if (!ScopeOrNone) {
    reportUnsupported(MI, "Unsupported atomic synchronization scope");
    return None;
  }

  OPUAtomicScope Scope = OPUAtomicScope::NONE;
  OPUAtomicAddrSpace OrderingAddrSpace = OPUAtomicAddrSpace::NONE;
  bool IsCrossAddressSpaceOrdering = false;
  std::tie(Scope, OrderingAddrSpace, IsCrossAddressSpaceOrdering) =
    ScopeOrNone.getValue();

  if ((OrderingAddrSpace == OPUAtomicAddrSpace::NONE) ||
      ((OrderingAddrSpace & OPUAtomicAddrSpace::ATOMIC) != OrderingAddrSpace)) {
    reportUnsupported(MI, "Unsupported atomic address space");
    return None;
  }

  return OPUMemOpInfo(Ordering, Scope, OrderingAddrSpace, OPUAtomicAddrSpace::ATOMIC,
                     IsCrossAddressSpaceOrdering);
}

Optional<OPUMemOpInfo> OPUMemOpAccess::getAtomicCmpxchgOrRmwInfo(
    const MachineBasicBlock::iterator &MI) const {
  assert(MI->getDesc().TSFlags & OPUInstrFlags::maybeAtomic);

  if (!(MI->mayLoad() && MI->mayStore()))
    return None;

  // Be conservative if there are no memory operands.
  if (MI->getNumMemOperands() == 0)
    return OPUMemOpInfo(AtomicOrdering::NotAtomic,
                        OPUAtomicScope::SYSTEM,
                        OPUAtomicAddrSpace::ATOMIC,
                        OPUAtomicAddrSpace::ALL,
                        AtomicOrdering::NotAtomic, false);
    // return OPUMemOpInfo();

  return constructFromMIWithMMO(MI);
}

OPUCacheControl::OPUCacheControl(const OPUSubtarget &ST) {
  TII = ST.getInstrInfo();
}

bool setCachePolicy(const MachineBasicBlock::iterator &MI, unsigned CachePolicy) const {
    // FIXME
    return false;
}


bool OPUCacheControl::enableLoadCacheBypass(
    const MachineBasicBlock::iterator &MI,
    OPUAtomicScope Scope,
    OPUAtomicAddrSpace AddrSpace) const {
  assert(MI->mayLoad() && !MI->mayStore());
  bool Changed = false;

  if ((AddrSpace & OPUAtomicAddrSpace::GLOBAL) != OPUAtomicAddrSpace::NONE) {
    /// TODO: Do not set glc for rmw atomic operations as they
    /// implicitly bypass the L1 cache.

    switch (Scope) {
    case OPUAtomicScope::SYSTEM:
    case OPUAtomicScope::AGENT:
      Changed |= setCachePolicy(MI, unsigned(OPUCacheBit::GLC));
      break;
    case OPUAtomicScope::BLOCK:
    case OPUAtomicScope::SINGLETHREAD:
      // No cache to bypass.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  /// The scratch address space does not need the global memory caches
  /// to be bypassed as all memory operations by the same thread are
  /// sequentially consistent, and no other thread can access scratch
  /// memory.

  /// Other address spaces do not hava a cache.

  return Changed;
}

bool OPUCacheControl::enableVolatile(
    const MachineBasicBlock::iterator &MI) const {
  assert(MI->mayLoad() ^ MI->mayStore());
  bool Changed = false;

  /// TODO: Do not enableGLCBit if rmw atomic.
  Changed |= OPUCacheBit(MI, unsigned(OPUCacheBit::GLC | OPUCacheBit::SLC));

  return Changed;
}

bool OPUCacheControl::insertCacheEvict(MachineBasicBlock::iterator &MI,
                                               OPUAtomicScope Scope,
                                               OPUAtomicAddrSpace AddrSpace,
                                               Position Pos) const {
  bool Changed = false;

  MachineBasicBlock &MBB = *MI->getParent();
  DebugLoc DL = MI->getDebugLoc();

  if (Pos == Position::AFTER)
    ++MI;

  if ((AddrSpace & OPUAtomicAddrSpace::GLOBAL) != OPUAtomicAddrSpace::NONE) {
    switch (Scope) {
    case OPUAtomicScope::SYSTEM:
    case OPUAtomicScope::DEVICE:
      BuildMI(MBB, MI, DL, TII->get(OPU::L1_WBINV_CLEAN);
      // FIXME BuildMI(MBB, MI, DL, TII->get(OPU::BUFFER_WBINVL1));
      Changed = true;
      break;
    case OPUAtomicScope::BLOCK:
    case OPUAtomicScope::SINGLETHREAD:
      // No cache to invalidate.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  /// The scratch address space does not need the global memory cache
  /// to be flushed as all memory operations by the same thread are
  /// sequentially consistent, and no other thread can access scratch
  /// memory.

  /// Other address spaces do not hava a cache.

  if (Pos == Position::AFTER)
    --MI;

  return Changed;
}

bool SIGfx6CacheControl::insertCacheInvalidate(MachineBasicBlock::iterator &MI,
                                               OPUAtomicScope Scope,
                                               OPUAtomicAddrSpace AddrSpace,
                                               Position Pos) const {
  bool Changed = false;

  MachineBasicBlock &MBB = *MI->getParent();
  DebugLoc DL = MI->getDebugLoc();

  if (Pos == Position::AFTER)
    ++MI;

  if ((AddrSpace & OPUAtomicAddrSpace::GLOBAL) != OPUAtomicAddrSpace::NONE) {
    switch (Scope) {
    case OPUAtomicScope::SYSTEM:
    case OPUAtomicScope::DEVICE:
      BuildMI(MBB, MI, DL, TII->get(OPU::L1_FLUSH));
      BuildMI(MBB, MI, DL, TII->get(OPU::S_WAIT_L1_WBINV));
      Changed = true;
      break;
    case OPUAtomicScope::BLOCK:
    case OPUAtomicScope::SINGLETHREAD:
      // No cache to invalidate.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  /// The scratch address space does not need the global memory cache
  /// to be flushed as all memory operations by the same thread are
  /// sequentially consistent, and no other thread can access scratch
  /// memory.

  /// Other address spaces do not hava a cache.

  if (Pos == Position::AFTER)
    --MI;

  return Changed;
}

bool GPUCacheControl::insertWait(MachineBasicBlock::iterator &MI,
                                    OPUAtomicScope Scope,
                                    OPUAtomicAddrSpace AddrSpace,
                                    OPUMemOp Op,
                                    bool IsCrossAddrSpaceOrdering,
                                    Position Pos) const {
  bool Changed = false;

  MachineBasicBlock &MBB = *MI->getParent();
  DebugLoc DL = MI->getDebugLoc();

  if (Pos == Position::AFTER)
    ++MI;

  OPU::Waitcnt Wait;

  if ((AddrSpace & OPUAtomicAddrSpace::GLOBAL) != OPUAtomicAddrSpace::NONE) {
    switch (Scope) {
    case OPUAtomicScope::SYSTEM:
    case OPUAtomicScope::DEVICE:
      if ((Op & OPUMemOp::LOAD) != OPUMemOp::NONE) {
        Wait.LDCnt = 0;
      }
      if ((Op & OPUMemOp::STORE) != OPUMemOp::NONE) {
        Wait.STCnt = 0;
      }
      break;
    case OPUAtomicScope::BLOCK:
    case OPUAtomicScope::SINGLETHREAD:
      // The L1 cache keeps all memory operations in order for
      // wavefronts in the same work-group.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

#if 0
  if ((AddrSpace & OPUAtomicAddrSpace::SHARED) != OPUAtomicAddrSpace::NONE) {
    switch (Scope) {
    case OPUAtomicScope::SYSTEM:
    case OPUAtomicScope::DEVICE:
      // If no cross address space ordering then an LDS waitcnt is not
      // needed as LDS operations for all waves are executed in a
      // total global ordering as observed by all waves. Required if
      // also synchronizing with global/GDS memory as LDS operations
      // could be reordered with respect to later global/GDS memory
      // operations of the same wave.
      LGKMCnt |= IsCrossAddrSpaceOrdering;
      /*
      SMCnt = IsCrossAddrSpaceOrdering;
      LMCnt = IsCrossAddrSpaceOrdering;
      MSGCnt = IsCrossAddrSpaceOrdering;
      */
      break;
    case OPUAtomicScope::WAVEFRONT:
    case OPUAtomicScope::SINGLETHREAD:
      // The LDS keeps all memory operations in order for
      // the same wavesfront.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }
#endif

#if 0
  if ((AddrSpace & OPUAtomicAddrSpace::GDS) != OPUAtomicAddrSpace::NONE) {
    switch (Scope) {
    case OPUAtomicScope::SYSTEM:
    case OPUAtomicScope::AGENT:
      // If no cross address space ordering then an GDS waitcnt is not
      // needed as GDS operations for all waves are executed in a
      // total global ordering as observed by all waves. Required if
      // also synchronizing with global/LDS memory as GDS operations
      // could be reordered with respect to later global/LDS memory
      // operations of the same wave.
      LGKMCnt |= IsCrossAddrSpaceOrdering;
      // EXPCnt = IsCrossAddrSpaceOrdering;
      break;
    case OPUAtomicScope::WORKGROUP:
    case OPUAtomicScope::WAVEFRONT:
    case OPUAtomicScope::SINGLETHREAD:
      // The GDS keeps all memory operations in order for
      // the same work-group.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }
#endif

    /* FIXME
  // if (VMCnt || SMCnt || LMCnt || MSGCnt) {
  if (VMCnt || LGKMCnt) {
    unsigned WaitCntImmediate =
      OPU::encodeWaitcnt(IV,
                            VMCnt ? 0 : getVmcntBitMask(IV),
                            getExpcntBitMask(IV),
                            LGKMCnt ? 0 : getLgkmcntBitMask(IV));
                            SMCnt ? 0 : getSmcntBitMask(IV),
                            LMCnt ? 0 : getLmcntBitMask(IV),
                            MSGCnt ? 0 : getMsgcntBitMask(IV),
                            getPbcntBitMask(IV));
    // BuildMI(MBB, MI, DL, TII->get(OPU::SL_WAIT)).addImm(WaitCntImmediate);
    BuildMI(MBB, MI, DL, TII->get(OPU::S_WAITCNT)).addImm(WaitCntImmediate);
    Changed = true;
  }
                            */

  if (Pos == Position::AFTER)
    --MI;

  return Changed;
}

bool OPUCacheControl::insertFence(MachineBasicBlock::iterator &MI,
                                               OPUAtomicScope Scope,
                                               OPUAtomicAddrSpace AddrSpace,
                                               OPUMemOp Op,
                                               Position Pos) const {
  bool Changed = false;

  MachineBasicBlock &MBB = *MI->getParent();
  DebugLoc DL = MI->getDebugLoc();

  if (Pos == Position::AFTER)
    ++MI;

  bool FenceSYS = false;
  bool FenceBLK = false;

  if ((AddrSpace & OPUAtomicAddrSpace::GLOBAL) != OPUAtomicAddrSpace::NONE) {
    switch (Scope) {
    case OPUAtomicScope::SYSTEM:
      if ((Op & OPUMemOp::STORE) != OPUMemOp::NONE) {
        FenceSYS = true;
      }
      break;
    case OPUAtomicScope::DEVICE:
      // FIXME BuildMI(MBB, MI, DL, TII->get(Flush));
      // BuildMI(MBB, MI, DL, TII->get(OPU::ML_LSA_WOPUNV));
      Changed = true;
      break;
    case OPUAtomicScope::BLOCK:
        FenceBLK = true;
    case OPUAtomicScope::SINGLETHREAD:
      // No cache to invalidate.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  if (FenceSYS) {
    BuildMI(MBB, MI, DL, TII->get(OPU::FENCE_SYS));
  } else if (FenceBLK) {
    OPU::Waitcnt Wait;
    uint64_t NewEnc = OPU::encodeWaitcnt(Wait);
    BuildMI(MBB, MI, DL, TII->get(OPU::FENCE_BLK));
    BuildMI(MBB, MI, DL, TII->get(OPU::S_WAIT_FENCE_BLK));
    BuildMI(MBB, MI, DL, TII->get(OPU::S_WAIT)).addImm(NewEnc);
  }

  /// The scratch address space does not need the global memory cache
  /// to be flushed as all memory operations by the same thread are
  /// sequentially consistent, and no other thread can access scratch
  /// memory.

  /// Other address spaces do not hava a cache.

  if (Pos == Position::AFTER)
    --MI;

  return Changed;
}

bool OPUCacheControl::insertWaitFence(MachineBasicBlock::iterator &MI,
                                               OPUAtomicScope Scope,
                                               OPUAtomicAddrSpace AddrSpace,
                                               OPUMemOp Op,
                                               Position Pos) const {
  bool Changed = false;

  MachineBasicBlock &MBB = *MI->getParent();
  DebugLoc DL = MI->getDebugLoc();

  if (Pos == Position::AFTER)
    ++MI;

  bool FenceSYS = false;

  if ((AddrSpace & OPUAtomicAddrSpace::GLOBAL) != OPUAtomicAddrSpace::NONE) {
    switch (Scope) {
    case OPUAtomicScope::SYSTEM:
      if ((Op & OPUMemOp::STORE) != OPUMemOp::NONE) {
        FenceSYS = true;
      }
      break;
    case OPUAtomicScope::DEVICE:
    case OPUAtomicScope::BLOCK:
    case OPUAtomicScope::SINGLETHREAD:
      // No cache to invalidate.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  if (FenceSYS) {
    BuildMI(MBB, MI, DL, TII->get(OPU::FENCE_SYS));
  }

  if (Pos == Position::AFTER)
    --MI;

  return Changed;
}

OPUSyncControl::OPUSyncControl(const OPUSubtarget &ST) {
  TII = ST.getInstrInfo();
}

void OPUSyncControl::analysisBlkSyn() {
  if (CrossScopeMIs.empty())
    return;

  for (auto CBB : CrossScopeMBBs) {
    MachineBasicBlock *MBB = CBB.first;
    if (BlkSyncMBBs.count(MBB)) {
      for (MachineBasicBlock::iterator I = MBB->end(); I != MBB-end(); I++) {
        if (BlkSyncMIs.count(&*I)) {
          CrossScopeMBBs[MBB] = false;
          break;
        } else if (CrossScopeMIs.count(&*I)) {
          break;
        }
      }
    }
  }

  // check whether blksync will reach CrossScopeMI before another blksyn
  for (auto BS : BlkSynMIs) {
    MachineBasicBlock *MBB = BS.first->getParent();
    // check cross scope MI in same block
    MachineBasicBlock::iterator I = BS.first;
    bool found = false;
    for (I++; I != MBB->end(); I++) {
      // find aother BlkSynMIs before CrossScopeMI
      if (BlkSynMIs.count(&*I)) {
        LLVM_DEBUG(dbgs() << "Find a light blksyn in MBB"
                          << BS.first->getParent()->getNumber()
                          << ", another blksyn in same block\n");
        found = true;
        break;
      } else if (CrossScopeMIs.count(&*I)) {
        BlkSynMIs[BS.first] = true;
        LLVM_DEBUG(dbgs() << "Find a strong blksyn in MBB"
                          << BS.first->getParent()->getNumber()
                          << ", CrossScopeMIs in same block\n");
        found = true;
        break;
      }
    }
    if (!found)
      BlkSynMIs[BS.first] = reachSuccCrossScopeMBB(BS.first);
  }
}

bool OPUSyncControl::reachSuccCrossScopeMBB(MachineInstr *MI) {
  SmallDenseMap<MachineBasicBlock *, bool> Checked;
  SmallVector<MachineBasicBlock *, 4> Stack;
  Stack.push_back(MI->getParent());

  LLVM_DEBUG(dbgs() << "Scan blksync in MBB"
                    << MI->getParent()->getNumber() << '\n');
  while(!Stack.empty()) {
    MachineBasicBlock *MBB = Stack.pop_back_val();
    for (MachineBasicBlock *Succ : MBB->successors()) {
      if (!Checked.count(Succ)) {
        LLVM_DEBUG(dbgs() << "Check SUCC = MBB" << Succ->getNumber() << '\n');
        // check whether Succ is CrossScopeMBB
        if (CrossScopeMBBs.count(Succ)) {
          if (CrossScopeMBBs[Succ]) {
            LLVM_DEBUG(dbg() << "It's a strong blksyn\n"
                             << "Succ CrossScopeMBBs is MBB"
                             <<  Succ->getNumber() << '\n');
            return true;
          }
        } else if (BlkSynMBBs.count(Succ)) {
          LLVM_DEBUG(dbg() << "It's a strong blksyn\n"
                             <<  Succ->getNumber() << '\n');
        } else {
          stack.push_back(Succ);
        }
        Checked.insert(std::make_pair(Succ, true));
      }
    }
  }
  LLVM_DEBUG(dbgs() << "It's a light blksync\n");
  return false;
}

bool OPUSyncControl::updateBlkSync() {
  bool Changed = false;
  for (auto BS : BlkSynMIs) {
    auto MBB = BS.first->getParent();
    if (BS.second) {
      DebugLoc DL = BS.first->getDebugLoc();
      OPU::Waitcnt Wait;
      Wait.LDCnt = 0;
      Wait.STCnt = 0;
      uint64_t NewEnc = OPU::encodeWaitcnt(Wait);
      BuildMI(*MBB, BS.first, DL, TII->get(OPU::S_WAIT)).addImm(NewEnc);
      Changed = true;
    } else {
      DebugLoc DL = BS.first->getDebugLoc();
      OPU::Waitcnt Wait;
      uint64_t NewEnc = OPU::encodeWaitcnt(Wait);
      BuildMI(*MBB, BS.first, DL, TII->get(OPU::MEM_FENCE_BLK));
      BuildMI(*MBB, BS.first, DL, TII->get(OPU::S_WAIT_MEM_FENCE_BLK));
      BuildMI(*MBB, BS.first, DL, TII->get(OPU::S_WAIT).addImm(NewEnc));
    }
  }
  return Changed;
}


bool OPUMemoryLegalizer::removeAtomicPseudoMIs() {
  if (AtomicPseudoMIs.empty())
    return false;

  for (auto &MI : AtomicPseudoMIs)
    MI->eraseFromParent();

  AtomicPseudoMIs.clear();
  return true;
}

bool OPUMemoryLegalizer::expandLoad(const OPUMemOpInfo &MOI,
                                   MachineBasicBlock::iterator &MI) {
  assert(MI->mayLoad() && !MI->mayStore());

  bool Changed = false;

  if (MOI.isAtomic()) {
    if (MOI.getOrdering() != AtomicOrdering::Monotonic ||
        MOI.getOrdering() == OPUAtomicScope::SYSTEM ||
        MOI.getOrdering() == OPUAtomicScope::DEVICE) {
      SC->insertCrossScopeMIs(&*MI));
    }

    // load.acquire = load.relaxed + fence.acquire
    // load.seq_cst = fence.seq_cst + load.relaxed + fence.seq_cs
    if (MOI.getOrdering() == AtomicOrdering::Monotonic ||
        MOI.getOrdering() == AtomicOrdering::Acquire ||
        MOI.getOrdering() == AtomicOrdering::SequentiallyConsistent) {
      Changed |= CC->enableLoadCacheBypass(MI, MOI.getScope(),
                                           MOI.getOrderingAddrSpace());
    }

    // Fence.R + Fence.W
    if (MOI.getOrdering() == AtomicOrdering::SequentiallyConsistent) {
      Changed |= CC->insertWait(MI,  MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                OPUMemOp::LOAD | OPUMemOp::STORE,
                                Position::BEFORE);
      Changed |= CC->insertFence(MI,  MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                OPUMemOp::LOAD | OPUMemOp::STORE,
                                Position::BEFORE);
      Changed |= CC->insertWaitFence(MI,  MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                OPUMemOp::LOAD | OPUMemOp::STORE,
                                Position::BEFORE);
    }

    if (MOI.getOrdering() == AtomicOrdering::Acquire ||
        MOI.getOrdering() == AtomicOrdering::SequentiallyConsistent) {
      // Fence.R
      Changed |= CC->insertWait(MI, MOI.getScope(),
                                MOI.getInstrAddrSpace(),
                                OPUMemOp::LOAD,
                                Position::AFTER);
      Changed |= CC->insertFence(MI, MOI.getScope(),
                                MOI.getInstrAddrSpace(),
                                OPUMemOp::LOAD,
                                Position::AFTER);
      // Fence.Flush
      Changed |= CC->insertCacheInvalidate(MI, MOI.getScope(),
                                           MOI.getOrderingAddrSpace(),
                                           Position::AFTER);
      Changed |= CC->insertWaitFence(MI, MOI.getScope(),
                                     MOI.getInstrAddrSpace(),
                                     OPUMemOp::LOAD,
                                     Position::AFTER);
    }

    return Changed;
  }

  // Atomic instructions do not have the nontemporal attribute.
  if (MOI.isVolatile()) {
    Changed |= CC->enableVolatile(MI);
    return Changed;
  }

  return Changed;
}

bool OPUMemoryLegalizer::expandStore(const OPUMemOpInfo &MOI,
                                    MachineBasicBlock::iterator &MI) {
  assert(!MI->mayLoad() && MI->mayStore());

  bool Changed = false;

  if (MOI.isAtomic()) {
    if (MOI.getOrdering() != AtomicOrdering::Monotonic) {
      if (MOI.getScope() = OPUAtomicScope::SYSTEM ||
          MOI.getScope() = OPUAtomicScope::DEVICE) {
        SC->insertCrossScopeMIs(&*MI);
      }
    }


    if (MOI.getOrdering() == AtomicOrdering::Release ||
        MOI.getOrdering() == AtomicOrdering::SequentiallyConsistent)
      Changed |= CC->insertWait(MI, MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                OPUMemOp::LOAD | OPUMemOp::STORE,
                                Position::BEFORE);
      Changed |= CC->insertFence(MI, MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                OPUMemOp::LOAD | OPUMemOp::STORE,
                                Position::BEFORE);
      // Cache.Clean
      Changed |= CC->insertCacheEvict(MI, MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                Position::BEFORE);
      Changed |= CC->insertWaitFence(MI, MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                OPUMemOp::LOAD | OPUMemOp::STORE,
                                Position::BEFORE);

    return Changed;
  }

  // Atomic instructions do not have the nontemporal attribute.
  if (MOI.isVolatile()) {
    Changed |= CC->enableVolatile(MI);
    return Changed;
  }

  return Changed;
}

bool OPUMemoryLegalizer::expandAtomicFence(const OPUMemOpInfo &MOI,
                                          MachineBasicBlock::iterator &MI) {
  assert(MI->getOpcode() == OPU::ATOMIC_FENCE);

  AtomicPseudoMIs.push_back(MI);
  bool Changed = false;

  if (MOI.isAtomic()) {
    if (MOI.getScope() = OPUAtomicScope::SYSTEM ||
        MOI.getScope() = OPUAtomicScope::DEVICE) {
      SC->insertCrossScopeMIs(&*MI);
    }

    if (MOI.getOrdering() == AtomicOrdering::Acquire ||
        MOI.getOrdering() == AtomicOrdering::Release ||
        MOI.getOrdering() == AtomicOrdering::AcquireRelease ||
        MOI.getOrdering() == AtomicOrdering::SequentiallyConsistent) {
      // Fence.R or Fence.RW
      Changed |= CC->insertWait(MI, MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                MOI.getOrdering() == AtomicOrdering::Acquire ?
                                    OPUMemOp::LOAD : OPUMemOp::LOAD | OPUMemOp::STORE,
                                Position::BEFORE);

      Changed |= CC->insertFence(MI, MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                MOI.getOrdering() == AtomicOrdering::Acquire ?
                                    OPUMemOp::LOAD : OPUMemOp::LOAD | OPUMemOp::STORE,
                                Position::BEFORE);
    }

    if (MOI.getOrdering() == AtomicOrdering::Release)
      Changed |= CC->insertCacheEvict(MI, MOI.getScope(),
                                MOI.getInstrAddrSpace(),
                                Position::BEFORE);


    if (MOI.getOrdering() == AtomicOrdering::Acquire ||
        MOI.getOrdering() == AtomicOrdering::AcquireRelease ||
        MOI.getOrdering() == AtomicOrdering::SequentiallyConsistent) {
      // Cache.Flush
      Changed |= CC->insertCacheInvalidate(MI, MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                Position::BEFORE);
    }

    if (MOI.getOrdering() == AtomicOrdering::Acquire ||
        MOI.getOrdering() == AtomicOrdering::Release ||
        MOI.getOrdering() == AtomicOrdering::AcquireRelease ||
        MOI.getOrdering() == AtomicOrdering::SequentiallyConsistent) {
      Changed |= CC->insertWaitFence(MI, MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                MOI.getOrdering() == AtomicOrdering::Acquire ?
                                    OPUMemOp::LOAD : OPUMemOp::LOAD | OPUMemOp::STORE,
                                Position::BEFORE);
    }

    return Changed;
  }

  return Changed;
}

bool OPUMemoryLegalizer::expandAtomicCmpxchgOrRmw(const OPUMemOpInfo &MOI,
  MachineBasicBlock::iterator &MI) {
  assert(MI->getOpcode() == OPU::ATOMIC_FENCE);

  AtomicPseudoMIs.push_back(MI);
  bool Changed = false;

  if (MOI.isAtomic()) {
    if (MOI.getOrdering() != AtomicOrdering::Monotonic ||
        MOI.getFailureOrdering() != AtomicOrdering::Monotonic) {
      if (MOI.getScope() = OPUAtomicScope::SYSTEM ||
          MOI.getScope() = OPUAtomicScope::DEVICE) {
        SC->insertCrossScopeMIs(&*MI);
      }
    }


    // atomic.acquire = atomic.relaxed + fence.acquire
    // atomic.release = fence.release + atomic.relaxed
    // atomic.acq_rel = fence.release + atomic.relaxed + fence.acquire
    // atomic.seq_cst = fence.seq_cst + atomic.relaxed + fence.seq_cst
    if (MOI.getOrdering() == AtomicOrdering::Release ||
        MOI.getOrdering() == AtomicOrdering::AcquireRelease ||
        MOI.getOrdering() == AtomicOrdering::SequentiallyConsistent ||
        MOI.getFailureOrdering() == AtomicOrdering::SequentiallyConsistent) {
      // Fence.RW
      Changed |= CC->insertWait(MI, MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                OPUMemOp::LOAD | OPUMemOp::STORE,
                                Position::BEFORE);

      Changed |= CC->insertFence(MI, MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                OPUMemOp::LOAD | OPUMemOp::STORE,
                                Position::BEFORE);

      Changed |= CC->insertWaitFence(MI, MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                OPUMemOp::LOAD | OPUMemOp::STORE,
                                Position::BEFORE);
    }

    if (MOI.getOrdering() == AtomicOrdering::Acquire ||
        MOI.getOrdering() == AtomicOrdering::AcquireRelease ||
        MOI.getOrdering() == AtomicOrdering::SequentiallyConsistent ||
        MOI.getFailureOrdering() == AtomicOrdering::Acquire ||
        MOI.getFailureOrdering() == AtomicOrdering::SequentiallyConsistent) {
      // Fence.R/Fence.RW
      Changed |= CC->insertWait(MI, MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                isAtomicRet(*MI) ? OPUMemOp::LOAD : OPUMemOp::STORE,
                                Position::AFTER);

      Changed |= CC->insertFence(MI, MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                isAtomicRet(*MI) ? OPUMemOp::LOAD : OPUMemOp::STORE,
                                Position::AFTER);

      // Cache.Flush
      Changed |= CC->insertCacheInvalidate(MI, MOI.getScope(),
                                MOI.getOrderingAddrSpace(),
                                Position::AFTER);

      Changed |= CC->insertWaitFence(MI, MOI.getScope(),
                                MOI.getInstrAddrSpace(),
                                isAtomicRet(*MI) ? OPUMemOp::LOAD : OPUMemOp::STORE,
                                Position::AFTER);
    }

    return Changed;
  }

  return Changed;
}

bool OPUMemoryLegalizer::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  OPUMemOpAccess MOA(MF);
  CC = OPUCacheControl::create(MF.getSubtarget<OPUSubtarget>());
  SC = OPUSyncControl::create(MF.getSubtarget<OPUSubtarget>());
  TII = MF.getSubtarget<OPUSubtarget>().getInstrInfo();

  for (auto &MBB : MF) {
    for (auto MI = MBB.begin(); MI != MBB.end(); ++MI) {
      DebugLoc DL = MI->getDebugLoc();
      MachineBasicBlock::iterator AfterMI = MI;
      AfterMI++;
      if (MI->getOpcode() == OPU::RMT_SIGNAL_ADD_U32_IMM) {
        SC->insertCrossScopeMIs(&*MI);
        int ModIdx = OPU::getNamedOperandIdx(MI->getOpcode(), OPU::OpName::flushmod);
        assert(ModIdx != -1);
        bool Flush = MI->getOperand(ModIdx).getImm();
        // insert S_WAIT before SIGNAL
        OPU::Waitcnt Wait;
        Wait.LDCnt = 0;
        Wait.STCnt = 0;
        uint64_t NewEnc = OPU::encodeWaitcnt(Wait);
        BuildMI(MBB, MI, DL, TII->get(OPU::S_WAIT)).addImm(NewEnc);
        // insert cache flush
        if (Flush) {
          BuildMI(MBB, MI, DL, TII->get(OPU::L1_WBINV_INV);
          BuildMI(MBB, MI, DL, TII->get(OPU::L1_WBINV_FLUSH);
          BuildMI(MBB, MI, DL, TII->get(OPU::S_WAIT_L1_WBINV);
        }
        if (MI->getOpcode() == OPU::RMT_SIGNAL_ADD_U32_STRONG_IMM) {
          Wait.LDCnt = ~0u;
          Wait.STCnt = ~0u;
          NewEnc = OPU::encodeWaitcnt(Wait);
          BuildMI(MBB, AfterMI, DL, TII->get(OPU::S_WAIT)).addImm(NewEnc);
        }
      }
    }
  }

  for (auto &MBB : MF) {
    for (auto MI = MBB.begin(); MI != MBB.end(); ++MI) {
      if (!(MI->getDesc().TSFlags & OPUInstrFlags::maybeAtomic))
        continue;
      if (const auto &MOI = MOA.getLoadInfo(MI))
        Changed |= expandLoad(MOI.getValue(), MI);
      else if (const auto &MOI = MOA.getStoreInfo(MI))
        Changed |= expandStore(MOI.getValue(), MI);
      else if (const auto &MOI = MOA.getAtomicFenceInfo(MI))
        Changed |= expandAtomicFence(MOI.getValue(), MI);
      else if (const auto &MOI = MOA.getAtomicCmpxchgOrRmwInfo(MI))
        Changed |= expandAtomicCmpxchgOrRmw(MOI.getValue(), MI);
    }
  }

  for (auto &MBB : MF) {
    for (auto MI = MBB.begin(); MI != MBB.end(); ++MI) {
      if (MI->getOpcode() == OPU::S_BLKSYN_CNT_IMM_SREG ||
          MI->getOpcode() == OPU::S_BLKSYN_CNT_IMM_IMM ||
          MI->getOpcode() == OPU::S_BLKSYN_CNT_SREG_IMM ||
          MI->getOpcode() == OPU::S_BLKSYN_CNT_SREG ||
          MI->getOpcode() == OPU::S_BLKSYN_CNT_IMM_SREG ||
          MI->getOpcode() == OPU::S_BLKSYN_CNT_IMM_IMM ||
          MI->getOpcode() == OPU::S_BLKSYN_CNT_SREG_IMM ||
          MI->getOpcode() == OPU::S_BLKSYN_CNT_SREG) {
        SC->insertBlkSynMIs(&*MI);
      }
    }
  }

  SC->analysisBlkSyn();
  Changed |= SC->updateBlkSyn();

  Changed |= removeAtomicPseudoMIs();
  return Changed;
}

INITIALIZE_PASS(OPUMemoryLegalizer, DEBUG_TYPE, PASS_NAME, false, false)

char OPUMemoryLegalizer::ID = 0;
char &llvm::OPUMemoryLegalizerID = OPUMemoryLegalizer::ID;

FunctionPass *llvm::createOPUMemoryLegalizerPass() {
  return new OPUMemoryLegalizer();
}
