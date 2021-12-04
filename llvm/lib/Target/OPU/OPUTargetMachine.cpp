//===-- OPUTargetMachine.cpp - Define TargetMachine for OPU -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the info about OPU target spec.
//
//===----------------------------------------------------------------------===//

#include "OPUTargetMachine.h"
#include "OPU.h"
#include "OPUAliasAnalysis.h"
#include "OPUMacroFusion.h"
#include "OPUTargetObjectFile.h"
#include "OPUTargetTransformInfo.h"
#include "OPUSchedStrategy.h"
#include "OPUIterativeScheduler.h"
#include "OPUMachineScheduler.h"
#include "TargetInfo/OPUTargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Vectorize.h"
using namespace llvm;

// Option to use reconverging CFG
///* FIXME schi we use feature instead of option
// MF->getSubtarget<OPUSubtarget>()->enableRevonverCFG();
static cl::opt<bool, true> ReconvergeCFG(
  "ppu-reconverge",
  cl::desc("Use reconverging CFG instead of structurization"),
  cl::location(OPUTargetMachine::EnableReconvergeCFG),
  cl::Hidden);

static cl::opt<bool> EnableSROA(
  "ppu-sroa",
  cl::desc("Run SROA after promote alloca pass"),
  cl::ReallyHidden,
  cl::init(true));

static cl::opt<bool> EnableEarlyIfConversion(
  "ppu-early-ifcvt",
  cl::Hidden,
  cl::desc("Run early if-conversion"),
  cl::init(false));

static cl::opt<bool> OptExecMaskPreRA(
  "ppu-opt-exec-mask-pre-ra", cl::Hidden,
  cl::desc("Run pre-RA exec mask optimizations"),
  cl::init(true));

// Option to disable vectorizer for tests.
static cl::opt<bool> EnableLoadStoreVectorizer(
  "ppu-load-store-vectorizer",
  cl::desc("Enable load store vectorizer"),
  cl::init(true),
  cl::Hidden);

// Option to control global loads scalarization
static cl::opt<bool> ScalarizeGlobal(
  "ppu-scalarize-global-loads",
  cl::desc("Enable global load scalarization"),
  cl::init(true),
  cl::Hidden);

// Option to run internalize pass.
static cl::opt<bool> InternalizeSymbols(
  "ppu-internalize-symbols",
  cl::desc("Enable elimination of non-kernel functions and unused globals"),
  cl::init(false),
  cl::Hidden);

// Option to inline all early.
static cl::opt<bool> EarlyInlineAll(
  "ppu-early-inline-all",
  cl::desc("Inline all functions early"),
  cl::init(false),
  cl::Hidden);

// Enable address space based alias analysis
static cl::opt<bool> EnableOPUAliasAnalysis("enable-ppu-aa", cl::Hidden,
  cl::desc("Enable OPU Alias Analysis"),
  cl::init(true));

// Option to run late CFG structurizer
static cl::opt<bool, true> LateCFGStructurize(
  "ppu-late-structurize",
  cl::desc("Enable late CFG structurization"),
  cl::location(OPUTargetMachine::EnableLateStructurizeCFG),
  cl::Hidden);


static cl::opt<bool, true> EnableOPUFunctionCallsOpt(
  "ppu-function-calls",
  cl::desc("Enable OPU function call support"),
  cl::location(OPUTargetMachine::EnableFunctionCalls),
  cl::init(true),
  cl::Hidden);

// Enable lib calls simplifications
static cl::opt<bool> EnableLibCallSimplify(
  "ppu-simplify-libcall",
  cl::desc("Enable ppu library simplifications"),
  cl::init(true),
  cl::Hidden);

static cl::opt<bool> EnableLowerKernelArguments(
  "ppu-ir-lower-kernel-arguments",
  cl::desc("Lower kernel argument loads in IR pass"),
  cl::init(true),
  cl::Hidden);

// TODO schi change to false
static cl::opt<bool> EnableRegReassign(
  "ppu-reassign-regs",
  cl::desc("Enable register reassign optimizations on gfx10+"),
  cl::init(false),
  cl::Hidden);

// Option is used in lit tests to prevent deadcoding of patterns inspected.
static cl::opt<bool>
EnableDCEInRA("ppu-dce-in-ra",
    cl::init(true), cl::Hidden,
    cl::desc("Enable machine DCE inside regalloc"));

static cl::opt<bool> EnableScalarIRPasses(
  "ppu-scalar-ir-passes",
  cl::desc("Enable scalar IR passes"),
  cl::init(true),
  cl::Hidden);

static cl::opt<bool, true> EnableSimtBranch(
  "ppu-simt-branch",
  cl::desc("Enable simt branch"),
  cl::location(OPUTargetMachine::EnableSimtBranch),
  cl::init(false),
  cl::Hidden);

static cl::opt<bool> EnableReorderBlocks(
  "ppu-reorder-block",
  cl::desc("Enable reorder-block for simt branch test"),
  cl::location(OPUTargetMachine::EnableReorderBlocks),
  cl::init(false),
  cl::ReallyHidden);

static cl::opt<bool> EnableRestrict(
  "ppu-restrict",
  cl::desc("assume all kernel ptr is restrict ptr"),
  cl::init(false),
  cl::ReallyHidden);

static cl::opt<bool> EnableFixUninit(
  "ppu-fix-uninit",
  cl::desc("ENable fix uninitializeds pass"),
  cl::init(true),
  cl::Hidden);

static cl::opt<bool> EnableFoldLogicOp(
  "ppu-fold-logic-op",
  cl::desc("ENable fold logic op with compare to zero"),
  cl::init(true),
  cl::Hidden);

static cl::opt<bool> EnableCoIssue(
  "ppu-co-issue",
  cl::desc("ENable opu co-issue"),
  cl::init(true),
  cl::Hidden);

static cl::opt<bool> EnableMaxOccupancy(
  "ppu-max-occupancy",
  cl::desc("ENable opu max occupancy scheduler"),
  cl::init(true),

static cl::opt<bool> EnableGuardWAR(
  "ppu-guard-war",
  cl::desc("ENable opu guard war pre-ra"),
  cl::init(true),

static cl::opt<int> RegMargin(
  "ppu-register-margin",
  cl::desc("Error Margin in OPUMaxOccupancySchedStrategy"),
  cl::init(3),
  cl::ReallyHidden);

static cl::opt<int> RegPressureInc(
  "ppu-register-inc",
  cl::desc("RegPressureInc"),
  cl::init(17),
  cl::ReallyHidden);

static cl::opt<bool> EnableIterMaxOccupancy(
  "ppu-iter-max-occupancy",
  cl::desc("Enable opu Iterative max occuancy scheduler"),
  cl::init(false));

static cl::opt<bool> EnableIterMinReg(
  "ppu-iter-min-reg",
  cl::desc("Enable opu Iterative min register scheduler"),
  cl::init(false));

static cl::opt<bool> EnableIterILP(
  "ppu-iter-ilp",
  cl::desc("Enable opu Iterative ilp scheduler"),
  cl::init(false));

static cl::opt<bool> OptVGPRLiveRange(
  "ppu-opt-vgpr-liverange",
  cl::desc("Enable VGPR liverange optimzations for if-else structure"),
  cl::init(true), cl::Hidden);

static cl::opt<bool> OptWaitCnt(
  "ppu-opt-waitcnt",
  cl::desc("Enable waitcnt optimization for if-else structure"),
  cl::init(true), cl::Hidden);

static cl::opt<bool, true> EnableOperandReuse(
  "ppu-operand-reuse",
  cl::desc("Enable opu VALU operand reuse"),
  cl::location(OPUTargetMachine::EnableOperandReuse),
  cl::init(true));

static cl::opt<bool> EnableEarlyIRTransform(
  "early-ir-tranform",
  cl::desc("Enable opu early IR transform"),
  cl::init(true), cl::Hidden);


extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeOPUTarget() {
  RegisterTargetMachine<OPUTargetMachine> X(getTheOPUTarget());

  //
  auto PR = PassRegistry::getPassRegistry();
  initializeGlobalISel(*PR);
  initializeOPUDAGToDAGISelPass(*PR);  //
  initializeOPUExpandPseudoPass(*PR);
  initializeOPUFoldOperandsPass(*PR);  //
  initializeOPULoadStoreOptimizerPass(*PR);
  initializeOPULowerSGPRSpillsPass(*PR);
  initializeOPULowerI1CopiesPass(*PR); //
  initializeOPUFixSGPRCopiesPass(*PR); //
  initializeOPUFixVGPRCopiesPass(*PR); //
  initializeOPUFixupVectorISelPass(*PR);
  initializeOPUAlwaysInlinePass(*PR);    // 
  initializeOPUAnnotateKernelFeaturesPass(*PR); //
  initializeOPUAnnotateUniformValuesPass(*PR);
  initializeOPUArgumentUsageInfoPass(*PR);  // 
  initializeOPULowerControlFlowPass(*PR);  //
  initializeOPULowerKernelArgumentsPass(*PR);
  initializeOPULowerKernelAttributesPass(*PR);
  initializeOPULowerIntrinsicsPass(*PR);
  initializeOPULowerReconvergingControlFlowPass(*PR);
  initializeOPUOptimizeExecMaskingPreRAPass(*PR);  //
  initializeOPUOptimizeExecMaskingPass(*PR);
  initializeOPUPromoteAllocaPass(*PR);  //
  initializeOPUPreAllocateWWMRegsPass(*PR);
  initializeOPUCodeGenPreparePass(*PR);
  // initializeOPUPropagateAttributesEarlyPass(*PR);
  // initializeOPUPropagateAttributesLatePass(*PR);
  initializeOPURewriteOutArgumentsPass(*PR);
  initializeOPUUnifyMetadataPass(*PR);
  initializeOPUAnnotateControlFlowPass(*PR);   //
  // initializeOPUInsertWaitcntsPass(*PR);     //
  initializeOPUUnifyDivergentExitNodesPass(*PR);  //  
  initializeOPUAAWrapperPassPass(*PR);       //
  initializeOPUExternalAAWrapperPass(*PR);   //
  initializeOPUInlinerPass(*PR);


  // InsertSkipPass ?
  // SmemSink ?
  // SGPRSpills?
  // initializeOPUAnnotateBlockDivergencyPass(*PR);   //
  // initializeOPUAsyncCopyAnalysisPass(*PR);   //
  // initializeOPUReorderBlocks(*PR);   //
  // initializeOPUForceSetCachePolicy(*PR);   //
  // initializeOPULowerAllocaPass(*PR);   //
  // initializeOPUSwitchICmpExtend(*PR);   //
  // initializeOPUAnnotateSmems(*PR);   //
  // initializeOPUAnnotateM0Pattern(*PR);   //
  // initializeLoadStoreOptimize(*PR);   //
  // initializeMMAExtendLiveRangePass(*PR);   //
  // initializeHandleKernelFunctionProloguePass(*PR);   //
  // initializeFixUninitializesPass(*PR);   //
  // initializeRecordUndefRegsPass(*PR);   //
  initializeOPUPreRAFusionPass(*PR);  //
  initializeOPUEarlyCodeGenPreparePass(*PR);
  initializeOPUAtomicOptimizerPass(*PR);
  initializeOPUWarpSyncInlinePass(*PR);
  initializeOPUOptimizeVGPRLiveRangePass(*PR);
  initializeOPUOptimizeWaitcntPass(*PR);
  initializeOPUDepResolverPass(*PR);
  initializeOPUMemoryLegalizerPass(*PR);
  initializeOPUResourceInfoPass(*PR);
  initializeOPUFoldLogicOpPass(*PR);  //
  // initializeOPUCoIssuePass(*PR);  //
  //initializeOPUPreAllocateFoldPass(*PR);  //
  //initializeOPUOperandReusePass(*PR);  //
  //initializeOPUEarlyIRTransformPass(*PR);  //
  //initializeOPURemoveGuardRegsPass(*PR);  //
}


static ScheduleDAGInstrs *createOPUMachineScheduler(MachineSchedContext *C) {
  return new OPUScheduleDAGMI(C);
}

static ScheduleDAGInstrs * createOPUMaxOccupancyMachineScheduler(MachineSchedContext *C) {
  ScheduleDAGMILive *DAG =
    new OPUScheduleDAGMILive(C, std::make_unique<OPUMaxOccupancySchedStrategy>(C, 
                RegMargin, RegPressureInc, EnableGuardWAR));
  DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
  DAG->addMutation(createStoreClusterDAGMutation(DAG->TII, DAG->TRI));
  DAG->addMutation(createOPUMacroFusionDAGMutation());
  return DAG;
}

static ScheduleDAGInstrs *
createIterativeOPUMaxOccupancyMachineScheduler(MachineSchedContext *C) {
  auto DAG = new OPUIterativeScheduler(C,
    OPUIterativeScheduler::SCHEDULE_LEGACYMAXOCCUPANCY,
    RegMargin, RegPressureInc);
  DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
  DAG->addMutation(createStoreClusterDAGMutation(DAG->TII, DAG->TRI));
  return DAG;
}

static ScheduleDAGInstrs *createMinRegScheduler(MachineSchedContext *C) {
  return new OPUIterativeScheduler(C,
    OPUIterativeScheduler::SCHEDULE_MINREGFORCED,
    RegMargin, RegPressureInc);
}

static ScheduleDAGInstrs *
createIterativeILPMachineScheduler(MachineSchedContext *C) {
  auto DAG = new OPUIterativeScheduler(C,
    OPUIterativeScheduler::SCHEDULE_ILP,
    RegMargin, RegPressureInc);
  DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
  DAG->addMutation(createStoreClusterDAGMutation(DAG->TII, DAG->TRI));
  DAG->addMutation(createOPUMacroFusionDAGMutation());
  return DAG;
}

static MachineSchedRegistry OPUSchedRegistry("ppu", "Run OPU's custom scheduler",
                createOPUMachineScheduler);

static MachineSchedRegistry OPUMaxOccupancySchedRegistry("ppu-max-occupancy",
                             "Run OPU scheduler to maximize occupancy",
                             createOPUMaxOccupancyMachineScheduler);

// TODO kernel compiled with Max/Min/ILP ,and runtime pps select best one to run
static MachineSchedRegistry IterativeOPUMaxOccupancySchedRegistry(
        "ppu-max-occupancy-experimental",
        "Run OPU scheduler to maximize occupancy (experimental)",
        createIterativeOPUMaxOccupancyMachineScheduler);

static MachineSchedRegistry OPUMinRegSchedRegistry(
        "ppu-minreg",
        "Run OPU iterative scheduler for minimal register usage (experimental)",
        createMinRegScheduler);

static MachineSchedRegistry OPUILPSchedRegistry(
        "ppu-ilp",
        "Run OPU iterative scheduler for ILP scheduling (experimental)",
        createIterativeILPMachineScheduler);

static StringRef computeDataLayout(const Triple &TT) {
  // odl riscv return "e-m:e-p:32:32-i64:64-n32-S128";
  // TODO OPU
  // 
  // 32-bit private, shared, and region pointers. 64-bit global, constant and
  // flat, non-integral buffer fat pointers.
    return "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
         "-i64:64-f16:16-f32:32-v16:16-v24:32-v32:32-v48:64-v96:128"
         "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"
         "-ni:7";
}

LLVM_READNONE
static StringRef getCPUOrDefault(const Triple &TT, StringRef GPU) {
  if (!GPU.empty())
    return GPU;

  // Need to default to a target with flat support for HSA.
  assert(TT.getArch() == Triple::ppu);
  // assert(TT.getOS() == Triple::PPS);
  return "opu";
}

static Reloc::Model getEffectiveRelocModel(const Triple &TT, Optional<Reloc::Model> RM) {
  if (!RM.hasValue())
    return Reloc::Static;
  // return *RM;
  return Reloc::PIC_;
}

OPUTargetMachine::OPUTargetMachine(const Target &T, const Triple &TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Optional<Reloc::Model> RM,
                                       Optional<CodeModel::Model> CM,
                                       CodeGenOpt::Level OL, bool JIT, bool EnRestrict)
    : LLVMTargetMachine(T, computeDataLayout(TT), TT, getCPUOrDefault(TT, CPU),
                        FS, Options, getEffectiveRelocModel(TT, RM),
                        getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<OPUELFTargetObjectFile>())
      // Options(Options)
      Subtarget(TT, CPU, FS, Options.MCOptions.getABIName(), *this), EnRestrict(EnableRestrict) {
  setSimtBranch(EnableSimtBranch);
  setReorderBlock(EnableReorderBlocks);
  initAsmInfo();

  // TODO this option will impact splitCriticalEdge check , tail merge checkit is not tested
  // if it will impact performance(also on SIMD)
  // which for SIMT, it is abuot correctness, so first trun on only for SIMT, 
  if (EnableSimtBranch)
    setRequiresStructuredCFG(true);
}

bool OPUTargetMachine::EnableReconvergeCFG = true;
bool OPUTargetMachine::EnableLateStructurizeCFG = false;
bool OPUTargetMachine::EnableFunctionCalls = false;
bool OPUTargetMachine::EnableSimtBranch = false;
bool OPUTargetMachine::EnableReorderBlocks = false;
bool OPUTargetMachine::EnableOperandReuse = true;

StringRef OPUTargetMachine::getFeatureString(const Function &F) const {
  Attribute FSAttr = F.getFnAttribute("target-features");

  return FSAttr.hasAttribute(Attribute::None) ?
    getTargetFeatureString() :
    FSAttr.getValueAsString();
}

/// Predicate for Internalize pass.
static bool mustPreserveGV(const GlobalValue &GV) {
  if (const Function *F = dyn_cast<Function>(&GV))
    return F->isDeclaration() || OPU::isEntryFunctionCC(F->getCallingConv());

  return !GV.use_empty();
}

void OPUTargetMachine::adjustPassManager(PassManagerBuilder &Builder) {
  Builder.DivergentTarget = true;

  bool EnableOpt = getOptLevel() >= CodeGenOpt::Default;

  bool Internalize = InternalizeSymbols;
  bool EarlyInline = EarlyInlineAll && EnableOpt && !EnableFunctionCalls;
  bool OPUAA = EnableOPUAliasAnalysis && EnableOpt;
  bool LibCallSimplify = EnableLibCallSimplify && EnableOpt;

  if (EnableFunctionCalls) {
    delete Builder.Inliner;
    Builder.Inliner = createOPUFunctionInliningPass();
  }

  Builder.addExtension(
    PassManagerBuilder::EP_ModuleOptimizerEarly,
    [Internalize, EarlyInline, OPUAA, this](const PassManagerBuilder &,
                                               legacy::PassManagerBase &PM) {
      if (OPUAA) {
        PM.add(createOPUAAWrapperPass());
        PM.add(createOPUExternalAAWrapperPass());
      }
      PM.add(createOPUUnifyMetadataPass());
      // TODO PM.add(createOPUPrintfRuntimeBinding());
      // TODO PM.add(createOPUPropagateAttributesLatePass(this));
      if (Internalize) {
        PM.add(createInternalizePass(mustPreserveGV));
        PM.add(createGlobalDCEPass());
      }
      if (EarlyInline)
        PM.add(createOPUAlwaysInlinePass(false));
  });

  const auto &Opt = Options;
  Builder.addExtension(
    PassManagerBuilder::EP_EarlyAsPossible,
    [OPUAA, LibCallSimplify, &Opt, this](const PassManagerBuilder &,
                                            legacy::PassManagerBase &PM) {
      if (OPUAA) {
        PM.add(createOPUAAWrapperPass());
        PM.add(createOPUExternalAAWrapperPass());
      }
      // TODO PM.add(llvm::createOPUPropagateAttributesEarlyPass(this));
      // TODO PM.add(llvm::createOPUUseNativeCallsPass());
      // TODO if (LibCallSimplify)
      // TODO   PM.add(llvm::createOPUSimplifyLibCallsPass(Opt, this));
  });

  Builder.addExtension(
    PassManagerBuilder::EP_CGSCCOptimizerLate,
    [](const PassManagerBuilder &, legacy::PassManagerBase &PM) {
      // Add infer address spaces pass to the opt pipeline after inlining
      // but before SROA to increase SROA opportunities.
      PM.add(createInferAddressSpacesPass());

      // This should run after inlining to have any chance of doing anything,
      // and before other cleanup optimizations.
      PM.add(createOPULowerKernelAttributesPass());
  });
}

const OPUSubtarget *OPUTargetMachine::getSubtargetImpl(const Function &F) const {
  StringRef CPU = getTargetCPU(); // "OPU";
  StringRef FS = getFeatureString(F);

  SmallString<128> SubtargetKey(CPU);
  SubtargetKey.append(FS);

  auto &I = SubtargetMap[SubtargetKey];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    I = std::make_unique<OPUSubtarget>(TargetTriple, CPU, FS, Options.MCOptions.getABIName(), *this);
  }

  I->setScalarizeGlobalBehavior(ScalarizeGlobal);

  return I.get();
}

TargetTransformInfo
OPUTargetMachine::getTargetTransformInfo(const Function &F) {
  return TargetTransformInfo(OPUTTIImpl(this, F));
}

namespace {
class OPUPassConfig : public TargetPassConfig {
public:
  OPUPassConfig(OPUTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM)
  {
    // Exceptions and StackMaps are not supported, so these passes will never do
    // anything.
    disablePass(&StackMapLivenessID);
    disablePass(&FuncletLayoutID);

    // It is necessary to know the register usage of the entire call graph.  We
    // allow calls without EnableOPUFunctionCalls if they are marked
    // noinline, so this is always required.
    setRequiresCodeGenSCCOrder(true);
  }

  OPUTargetMachine &getOPUTargetMachine() const {
    return getTM<OPUTargetMachine>();
  }

  ScheduleDAGInstrs *
  createMachineScheduler(MachineSchedContext *C) const override {
    /*
    const OPUSubtarget &ST = C->MF->getSubtarget<OPUSubtarget>();
    if (ST.enableOPUScheduler())
        return createOPUMachineScheduler(C);
    */
    if (EnableMaxOccupancy)
      return createOPUMaxOccupancyMachineScheduler(C);
    else if (EnableIterILP)
      return createIterativeILPMachineScheduler(C);
    else if (EnableIterMaxOccupancy)
      return createIterativeOPUMaxOccupancyMachineScheduler(C);
    else if (EnableIterMinReg)
      return createMinRegScheduler(C);

    return createOPUMaxOccupancyMachineScheduler(C);
  }

  // GCNBase
  void addEarlyCSEOrGVNPass();
  void addStraightLineScalarOptimizationPasses();

  void addIRPasses() override;
  void addCodeGenPrepare() override;
  bool addPreISel() override;
  bool addInstSelector() override;
  // bool addGCPasses() override;

  // GCN
  // bool addPreISel() override;
  void addMachineSSAOptimization() override;
  bool addILPOpts() override;
  // bool addInstSelector() override;
  bool addIRTranslator() override;
  bool addLegalizeMachineIR() override;

    return createOPUMaxOccupancyMachineScheduler(C);
  }

  // GCNBase
  void addEarlyCSEOrGVNPass();
  void addStraightLineScalarOptimizationPasses();
  void addIRPasses() override;
  void addCodeGenPrepare() override;
  bool addPreISel() override;
  bool addInstSelector() override;
  // bool addGCPasses() override;

  // GCN
  // bool addPreISel() override;
  void addMachineSSAOptimization() override;
  bool addILPOpts() override;
  // bool addInstSelector() override;
  bool addIRTranslator() override;
  bool addLegalizeMachineIR() override;
  bool addRegBankSelect() override;
  bool addGlobalInstructionSelect() override;

  void addFastRegAlloc() override;
  void addOptimizedRegAlloc() override;
  void addPreRegAlloc() override;
  bool addPreRewrite() override;
  void addPostRegAlloc() override;
  void addPreSched2() override;
  void addPreEmitPass() override;

  // RISCV
  // void addIRPasses() override;
  // bool addInstSelector() override;
  // bool addIRTranslator() override;
  // bool addLegalizeMachineIR() override;
  // bool addRegBankSelect() override;
  // bool addGlobalInstructionSelect() override;
  // void addPreEmitPass() override;
  void addPreEmitPass2() override;
  // void addPreRegAlloc() override;

  std::unique_ptr<CSEConfigBase> getCSEConfig() const override {
    return getStandardCSEConfigForOpt(TM->getOptLevel());
  }
};
}

void OPUPassConfig::addEarlyCSEOrGVNPass() {
  if (getOptLevel() == CodeGenOpt::Aggressive)
    addPass(createGVNPass());
  else
    addPass(createEarlyCSEPass());
}

void OPUPassConfig::addStraightLineScalarOptimizationPasses() {
  addPass(createLICMPass());
  addPass(createSeparateConstOffsetFromGEPPass());
  addPass(createSpeculativeExecutionPass());
  // ReassociateGEPs exposes more opportunites for SLSR. See
  // the example in reassociate-geps-and-slsr.ll.
  addPass(createStraightLineStrengthReducePass());
  // SeparateConstOffsetFromGEP and SLSR creates common expressions which GVN or
  // EarlyCSE can reuse.
  addEarlyCSEOrGVNPass();
  // Run NaryReassociate after EarlyCSE/GVN to be more effective.
  addPass(createNaryReassociatePass());
  // NaryReassociate on GEPs creates redundant common expressions, so run
  // EarlyCSE after it.
  addPass(createEarlyCSEPass());
}

void OPUPassConfig::addIRPasses() {
  const OPUTargetMachine &TM = getOPUTargetMachine();

  // There is no reason to run these.
  disablePass(&StackMapLivenessID);
  disablePass(&FuncletLayoutID);
  disablePass(&PatchableFunctionID);

  // TODO schi addPass(createOPUPrintfRuntimeBinding());
  // This must occur before inlining, as the inliner will not look through
  // bitcast calls.
  // TODO schi addPass(createOPUFixFunctionBitcastsPass());
  // A call to propagate attributes pass in the backend in case opt was not run.
  // TODO schi addPass(createOPUPropagateAttributesEarlyPass(&TM));
 
  // RISCV
  addPass(createAtomicExpandPass());

  addPass(createOPULowerIntrinsicsPass());   //
  addPass(createOPUEarlyCodeGenPreparePass());

  // TODO schi if (!EnableFunctionCalls) {
  // Function calls are not supported, so make sure we inline everything.
  addPass(createOPUAlwaysInlinePass(true));
  if (EnableSimtBranch)
    addPass(createOPUWarpSyncInlinePass(true));
  else
    addPass(createOPUWarpSyncInlinePass(false));
  addPass(createAlwaysInlinerLegacyPass());
  // We need to add the barrier noop pass, otherwise adding the function
  // inlining pass will cause all of the PassConfigs passes to be run
  // one function at a time, which means if we have a nodule with two
  // functions, then we will generate code for the first function
  // without ever running any passes on the second.
  addPass(createBarrierNoopPass());
  // TODO schi end if EnableFunctionCalls

  addPass(createOPUCodeGenPreparePass());

  // Replace OpenCL enqueued block function pointers with global variables.
  // TODO addPass(createOPUOpenCLEnqueuedBlockLoweringPass());

  if (TM.getOptLevel() > CodeGenOpt::None) {
    addPass(createInferAddressSpacesPass());
    addPass(createOPUPromoteAlloca());

    if (EnableSROA)
      addPass(createSROAPass());

    addPass(createOPULowerAllocaPass());
    addPass(createInferAddressSpacesPass());

    // split some generic atomic op which bsm not supported
    addPass(createOPUAtomicSpliterPass());

    if (EnableScalarIRPasses)
      addStraightLineScalarOptimizationPasses();


    if (EnableOPUAliasAnalysis) {
      addPass(createOPUAAWrapperPass());
      addPass(createExternalAAWrapperPass([](Pass &P, Function &,
                                             AAResults &AAR) {
        if (auto *WrapperPass = P.getAnalysisIfAvailable<OPUAAWrapperPass>())
          AAR.addAAResult(WrapperPass->getResult());
        }));
    }
  }

  // TODO: May want to move later or split into an early and late one.
  addPass(createOPUCodeGenPreparePass());
  addPass(createLICMPass(true));

  TargetPassConfig::addIRPasses();

  // EarlyCSE is not always strong enough to clean up what LSR produces. For
  // example, GVN can combine
  //
  //   %0 = add %a, %b
  //   %1 = add %b, %a
  //
  // and
  //
  //   %0 = shl nsw %a, 2
  //   %1 = shl %a, 2
  //
  // but EarlyCSE can do neither of them.
  if (getOptLevel() != CodeGenOpt::None && EnableScalarIRPasses)
    addEarlyCSEOrGVNPass();
}

void OPUPassConfig::addCodeGenPrepare() {
  if (TM->getTargetTriple().getArch() == Triple::ppu)
    addPass(createOPUAnnotateKernelFeaturesPass());

  if (TM->getTargetTriple().getArch() == Triple::ppu &&
      EnableLowerKernelArguments)
    addPass(createOPULowerKernelArgumentsPass());

  addPass(&OPUPerfHintAnalysisID);

  TargetPassConfig::addCodeGenPrepare();

  if (EnableLoadStoreVectorizer)
    addPass(createLoadStoreVectorizerPass());
}

bool OPUPassConfig::addPreISel() {
  // GCNBase
  addPass(createLowerSwitchPass());
  //addPass(createOPUSwitchICmpExtendPass());
  addPass(createFlattenCFGPass());

  if (!EnableReorderBlocks)
    addPass(createOPUAtomicOptimizerPass());

  if (!EnableM0Pattern)
    addPass(createOPUAnnoateM0PatternsPass());

  //addPass(createOPUAnnotateSmemPass());
  addPass(createOPUAnnotateUniformValues());

  // Merge divergent exit nodes. StructurizeCFG won't recognize the multi-exit
  // regions formed by them.
  addPass(&OPUUnifyDivergentExitNodesID);

  if (ReconvergeCFG) {
    addPass(createReconvergeCFGPass(true)); // true -> SkipUniformBranches
  } else if (!LateCFGStructurize) {
    addPass(createStructurizeCFGPass(true)); // true -> SkipUniformRegions
  }
  addPass(createSinkingPass());
  addPass(createOPUAnnotateUniformValues());
  if (!ReconvergeCFG && !LateCFGStructurize) {
    addPass(createOPUAnnotateControlFlowPass());
  }
  addPass(createLCSSAPass());
  addPass(createOPUAnnotateModeRegister());

  return false;
}

bool OPUPassConfig::addInstSelector() {
  // GCNBase
  // Defer the verifier until FinalizeISel.
  // addPass(createOPUISelDag(getOPUTargetMachine()));
  addPass(createOPUISelDag(getOPUTargetMachine(), getOptLevel()), false);

  // GCN
  addPass(&OPUFixSGPRCopiesID);
  addPass(createOPULowerI1CopiesPass());
  addPass(createOPUFixupVectorISelPass());
  // addPass(createSIAddIMGInitPass());
  // FIXME: Remove this once the phi on CF_END is cleaned up by either removing
  // LCSSA or other ways.
  addPass(&UnreachableMachineBlockElimID);
  return false;
}

void OPUPassConfig::addMachineSSAOptimization() {
  TargetPassConfig::addMachineSSAOptimization();

  // We want to fold operands after PeepholeOptimizer has run (or as part of
  // it), because it will eliminate extra copies making it easier to fold the
  // real source operand. We want to eliminate dead instructions after, so that
  // we see fewer uses of the copies. We then need to clean up the dead
  // instructions leftover after the operands are folded as well.
  //
  // XXX - Can we get away without running DeadMachineInstructionElim again?
  addPass(&OPUFoldOperandsID);
  if (EnableFoldLogicOp)
    addPass(&OPUFoldLogicOpID);
  /* TODO
  if (EnableDPPCombine)
    addPass(&GCNDPPCombineID);
  */
  addPass(&DeadMachineInstructionElimID);
  addPass(&OPULoadStoreOptimizerID);
  /* TODO
  if (EnableSDWAPeephole) {
    addPass(&SIPeepholeSDWAID);
    addPass(&EarlyMachineLICMID);
    addPass(&MachineCSEID);
    addPass(&SIFoldOperandsID);
    addPass(&DeadMachineInstructionElimID);
  }
  */
  // TODO schi addPass(createOPUShrinkInstructionsPass());
  
  // RISCV-V
  addPass(createOPUOptimizeVSETVLUsesPass());
}

bool OPUPassConfig::addILPOpts() {
  if (EnableEarlyIfConversion)
    addPass(&EarlyIfConverterID);

  TargetPassConfig::addILPOpts();
  return false;
}

bool OPUPassConfig::addIRTranslator() {
  addPass(new IRTranslator());
  return false;
}

bool OPUPassConfig::addLegalizeMachineIR() {
  addPass(new Legalizer());
  return false;
}

bool OPUPassConfig::addRegBankSelect() {
  addPass(new RegBankSelect());
  return false;
}

bool OPUPassConfig::addGlobalInstructionSelect() {
  addPass(new InstructionSelect());
  return false;
}

void OPUPassConfig::addFastRegAlloc() {
  // FIXME: We have to disable the verifier here because of PHIElimination +
  // TwoAddressInstructions disabling it.

  // This must be run immediately after phi elimination and before
  // TwoAddressInstructions, otherwise the processing of the tied operand of
  // SI_ELSE will introduce a copy of the tied operand source after the else.
  insertPass(&PHIEliminationID, &OPULowerControlFlowID, false);

  // This must be run just after RegisterCoalescing.
  //insertPass(&RegisterCoalescerID, &OPUPreAllocateWWMRegsID, false);
  insertPass(&RegisterCoalescerID, &OPUPreAllocateFoldID, false);

  // OPUMMAEtendLiveRangesID need insert after MachienScheduelerID
  //insertPass(&RegisterCoalescerID, &OPUMMAExtendLivRangesID, false);

  TargetPassConfig::addFastRegAlloc();
}

void OPUPassConfig::addOptimizedRegAlloc() {
  // char *PassID = &PHIEliminationID;

  if (OptExecMaskPreRA) {
    if (!ReconvergeCFG)
      insertPass(&MachineSchedulerID, &OPUOptimizeExecMaskingPreRAID);

    if (OptVGPRLiveRange)
      insertPass(&LiveVariablesID, &OPUOptimizeVGPRLiveRangeID, false);
    // insertPass(&OPUOptimizeExecMaskingPreRAID, &OPUFormMemoryClausesID);
  } else {
    // insertPass(&MachineSchedulerID, &OPUFormMemoryClausesID);
  }

  // This must be run immediately after phi elimination and before
  // TwoAddressInstructions, otherwise the processing of the tied operand of
  // SI_ELSE will introduce a copy of the tied operand source after the else.
  if (OptWaitCnt)
    insertPass(&LiveVariablesID, &OPUOptimizeWaitcntID, false);

  insertPass(&PHIEliminationID, &OPULowerControlFlowID, false);

  // This must be run just after RegisterCoalescing.
  //insertPass(&RegisterCoalescerID, &OPUPreAllocateWWMRegsID, false);
  insertPass(&RegisterCoalescerID, &OPUPreAllocateFoldID, false);
  insertPass(&MachineSchedulerID, &OPUMMAExtendLiveRangeID, false);

  if (EnableDCEInRA)
    insertPass(&RenameIndependentSubregsID, &DeadMachineInstructionElimID);

  TargetPassConfig::addOptimizedRegAlloc();
}

void OPUPassConfig::addPreRegAlloc() {
  if (LateCFGStructurize) {
    // addPass(createAMDGPUMachineCFGStructurizerPass());
  }
  if (ReconvergeCFG)
    addPass(createOPULowerReconvergingControlFlowPass());

  // TODO addPass(createSIWholeQuadModePass());

  // RISCV
  addPass(createOPUMergeBaseOffsetOptPass());
}

bool OPUPassConfig::addPreRewrite() {
  //if (EnableRegReassign) {
  //  // addPass(&OPUNSAReassignID);
  //  addPass(&OPURegBankReassignID);
  //}

  if (EnableFixUninit) {
    addPass(&OPURecordUndefRegsID);
  }
  return true;
}

void OPUPassConfig::addPostRegAlloc() {
  if (EnableGuardWAR)
    addPass(&OPURemoveGuardRegsID);

  addPass(&OPUFixVGPRCopiesID);

  if (!EnableSimtBranch) {
    if (getOptLevel() > CodeGenOpt::None)
      addPass(&OPUOptimizeExecMaskingID);
  }

  TargetPassConfig::addPostRegAlloc();

  // Equivalent of PEI for SGPRs.
  addPass(&OPULowerSGPRSpillsID);
}

void OPUPassConfig::addPreSched2() {
}

void OPUPassConfig::addPreEmitPass() {
  if (EnableCoIssue)
    addPass(&OPUCoIssueID);

  addPass(createOPUMemoryLegalizerPass());
  addPass(createOPUHandleKernelFunctionPrologue());
  addPass(createOPUInsertWaitcntsPass());
  addPass(createOPUDepResolverPass());

  if (EnableFixUnint)
    addPass(createOPUFixUninitializesPass());
  // TODO schi addPass(createOPUInsertWaitcntsPass());
  // TODO schi addPass(createSIShrinkInstructionsPass());
  // TODO schi addPass(createOPUModeRegisterPass());
  //
  //
  //if (ForceSetCachePolicy != 1) {
  //  addPass(CreateOPUForceSetCachePolicyPass)
  //}

  // The hazard recognizer that runs as part of the post-ra scheduler does not
  // guarantee to be able handle all hazards correctly. This is because if there
  // are multiple scheduling regions in a basic block, the regions are scheduled
  // bottom up, so when we begin to schedule a region we don't know what
  // instructions were emitted directly before it.
  //
  // Here we add a stand-alone hazard recognizer pass which can handle all
  // cases.
  //
  // FIXME: This stand-alone pass will emit indiv. S_NOP 0, as needed. It would
  // be better for it to emit S_NOP <N> when possible.
  addPass(&PostRAHazardRecognizerID);

  if (!EnableSimtBranch)
    addPass(&OPUInsertSkipsPassID);
  else if (EnalbeReorderBlocks) {
    //addPass(&OPUReorderBlocksPassID);
  }

  if (!EnableOperandReuse)
    addPass(&OPUOperandReusePassID);
  // FIXME addPass(&OPUInsertSkipsPassID);

  // RISCV & GCN
  addPass(&BranchRelaxationPassID);
}


// RISCV
void OPUPassConfig::addPreEmitPass2() {
  // Schedule the expansion of AMOs at the last possible moment, avoiding the
  // possibility for other passes to break the requirements for forward
  // progress in the LR/SC block.
  addPass(createOPUExpandPseudoPass());
}



// FIXME

TargetPassConfig *OPUTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new OPUPassConfig(*this, PM);
}


